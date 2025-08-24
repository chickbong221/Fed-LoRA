import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import numpy as np
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from torch.optim.lr_scheduler import StepLR


# TODO 1. Fix all of the value names of FedAvg -> FedExAgg
class ExactClientsAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None

        avg_model = self._grad_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        # print("chicken")
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _grad_weighted_avg(self, models, recover_fun=None):
        """
        Apply Exact LoRA scaling using U, V for each client's gradients,
        then calculate the weighted average and update the global model.
        """
        # Extract current parameters and gradients
        A_all, B_all = self.extract_lora_AB(models)
        grad_A_all, grad_B_all = self.extract_lora_AB_gradients(models)
        
        # Optimize U, V based on gradients to achieve ideal parameter update
        U, V = self.optimize_exact_uv(models, lr=1e-2, steps=200)
        a, b = self.optimize_conflict_free_uv(models)

        num_clients = len(models)
        training_set_size = sum(sample_size for sample_size, _ in models)

        # Start with global model as base
        avg_model = {}
        global_state_dict = self.model.state_dict()
        
        # Initialize avg_model with global model parameters
        for key in global_state_dict:
            avg_model[key] = global_state_dict[key].clone()

        # Compute weighted average of scaled gradients
        total_weighted_gradients = {}
        
        for key in global_state_dict:
            total_weighted_gradients[key] = torch.zeros_like(global_state_dict[key])
            
            for i in range(num_clients):
                local_sample_size, local_model = models[i]

                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / num_clients
                elif self.cfg.federate.use_ss:
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                # Compute gradient for this client and parameter
                if key in local_model:
                    if not self.cfg.federate.use_ss:
                        local_param = param2tensor(local_model[key])
                    else:
                        local_param = local_model[key]
                    
                    # Compute gradient (client_param - global_param)
                    gradient = local_param - global_state_dict[key]

                    # Scale gradient with U[i] or V[i] if key contains lora_A or lora_B
                    scaled_gradient = gradient
                    if "lora_A" in key:
                        scaled_gradient = U[i] * gradient
                    elif "lora_B" in key:
                        scaled_gradient = V[i] * gradient

                    # Accumulate weighted scaled gradients
                    total_weighted_gradients[key] += scaled_gradient * weight
                    # print(total_weighted_gradients[key].device)

            # Secret sharing post-processing for gradients
            if self.cfg.federate.use_ss and recover_fun:
                total_weighted_gradients[key] = recover_fun(total_weighted_gradients[key])
                total_weighted_gradients[key] /= training_set_size
                total_weighted_gradients[key] = torch.FloatTensor(total_weighted_gradients[key])

        # Apply accumulated gradients to global model
        for key in avg_model:
            avg_model[key] = global_state_dict[key] + total_weighted_gradients[key]

        return avg_model

    def optimize_exact_uv(self, models, lr=1e-2, steps=200):
 
        # Extract current parameters and gradients
        A_all, B_all = self.extract_lora_AB(models)
        grad_A_all, grad_B_all = self.extract_lora_AB_gradients(models)
        
        # Move to device
        A_all = A_all.to(self.device)
        B_all = B_all.to(self.device)
        grad_A_all = grad_A_all.to(self.device)
        grad_B_all = grad_B_all.to(self.device)
        
        num_clients = A_all.shape[0]
        
        # Get global model's current LoRA parameters
        global_state_dict = self.model.state_dict()
        global_A_dict = {}
        global_B_dict = {}
        
        for name, param in global_state_dict.items():
            if "lora_A" in name:
                global_A_dict[name] = param.detach().clone()
            elif "lora_B" in name:
                global_B_dict[name] = param.detach().clone()
        
        # Convert to same format as A_all, B_all (sorted order)
        sorted_A_names = sorted(global_A_dict.keys())
        sorted_B_names = sorted(global_B_dict.keys())
        
        global_A_list = [global_A_dict[name] for name in sorted_A_names]
        global_B_list = [global_B_dict[name] for name in sorted_B_names]
        
        global_A = torch.stack(global_A_list).to(self.device)  # [num_layers, ...]
        global_B = torch.stack(global_B_list).to(self.device)  # [num_layers, ...]
        
        # Compute ideal update from the original optimize_uv objective
        BA = []
        for i in range(len(A_all)):
            BA.append(torch.matmul(B_all[i], A_all[i]))
        ideal_update = torch.stack(BA).mean(dim=0)  # [num_layers, ...]
        
        # Initialize optimization parameters
        U = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        V = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        
        optimizer = torch.optim.AdamW([U, V], lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Apply weighted gradients to global parameters
            weighted_grad_A = (U.view(-1, 1, 1, 1) * grad_A_all).mean(dim=0)  # [num_layers, ...]
            weighted_grad_B = (V.view(-1, 1, 1, 1) * grad_B_all).mean(dim=0)  # [num_layers, ...]
            
            # Updated parameters after applying weighted gradients
            updated_A = global_A + weighted_grad_A  # [num_layers, ...]
            updated_B = global_B + weighted_grad_B  # [num_layers, ...]
            
            # Compute the update that would be achieved with these new parameters
            achieved_update = torch.matmul(updated_B, updated_A)  # [num_layers, ...]

            # Loss: difference between achieved update and ideal update
            ideal_update = ideal_update.float() 
            achieved_update = achieved_update.float()
            diff = (achieved_update - ideal_update) / (torch.min(achieved_update - ideal_update)+ 1e-8)

            loss = torch.mean(diff ** 2)

            loss.backward()
            optimizer.step()
            
            # if step % 10 == 0:
            #     print(f"Step {step}: Loss = {loss.item()}")
        
        return U.detach().squeeze().cpu(), V.detach().squeeze().cpu()  # shape: [num_clients]

    def optimize_conflict_free_uv(self, models):
        
        CA_lr = self.cfg.federate.FLoRA_CA_lr
        CA_momentum = self.cfg.federate.FLoRA_CA_momentum
        CA_step_size = self.cfg.federate.FLoRA_CA_step_size
        CA_gamma = self.cfg.federate.FLoRA_CA_gamma
        CA_c = self.cfg.federate.FLoRA_CA_c
        CA_grad_balance = self.cfg.federate.FLoRA_CA_grad_balance
        CA_step = self.cfg.federate.FLoRA_CA_step

        # Extract current parameters and gradients
        grad_A_all, grad_B_all = self.extract_lora_AB_gradients(models)

        # --- Gradient Normalization for A and B ---
        if CA_grad_balance:
            balanced_grad_A = []
            balanced_grad_B = []

            for i in range(grad_A_all.size(0)):  # loop over clients
                # Flatten A and B grads for this client
                grad_A_flat = grad_A_all[i].reshape(-1)
                grad_B_flat = grad_B_all[i].reshape(-1)

                # Compute combined norm
                grad_norm = torch.norm(torch.cat([grad_A_flat, grad_B_flat]))

                # Target norm = mean across all clients
                balanced_grad_A.append((grad_A_all[i] / grad_norm).unsqueeze(0))
                balanced_grad_B.append((grad_B_all[i] / grad_norm).unsqueeze(0))

            balanced_grad_A = torch.cat(balanced_grad_A, dim=0)
            balanced_grad_B = torch.cat(balanced_grad_B, dim=0)

            # Compute target scaling (mean of all client norms)
            all_norms = [torch.norm(torch.cat([grad_A_all[i].reshape(-1), 
                                            grad_B_all[i].reshape(-1)])) 
                        for i in range(grad_A_all.size(0))]
            target_norm = torch.mean(torch.stack(all_norms))

            # Rescale each client’s grads to target_norm
            scaling_factors = [target_norm / n if n > 0 else 1.0 for n in all_norms]

            grad_A_all = torch.stack([g * s for g, s in zip(balanced_grad_A, scaling_factors)])
            grad_B_all = torch.stack([g * s for g, s in zip(balanced_grad_B, scaling_factors)])
        
        # Move to device
        grad_A_all = grad_A_all.to(self.device)
        grad_B_all = grad_B_all.to(self.device)
        
        num_clients = grad_A_all.shape[0]

        # Compute ideal gradient
        BA = []
        for i in range(len(grad_A_all)):
            BA.append(torch.matmul(grad_B_all[i], grad_A_all[i]))
        ideal_gradient = torch.stack(BA).mean(dim=0)  # [num_layers, ...]

        # Initialize U and V weights
        U = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        V = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        
        # Setup optimizer for both U and V
        optimizer = torch.optim.SGD([U, V], lr=CA_lr, momentum=CA_momentum)
        scheduler = StepLR(optimizer, step_size=CA_step_size, gamma=CA_gamma)

        loss_best = np.inf
        U_best = None
        V_best = None

        for step in range(CA_step + 1):
            optimizer.zero_grad()
            
            weighted_grad_A = (U.view(-1, 1, 1, 1) * grad_A_all).mean(dim=0)  # [num_layers, ...]
            weighted_grad_B = (V.view(-1, 1, 1, 1) * grad_B_all).mean(dim=0)  # [num_layers, ...]
            achieved_gradient = torch.matmul(weighted_grad_B, weighted_grad_A)

            # Loss: difference between achieved update and ideal update
            ideal_gradient = ideal_gradient.float() 
            # ideal_gradient = ideal_gradient / torch.min(ideal_gradient)
            achieved_gradient = achieved_gradient.float()
            # achieved_gradient = achieved_gradient / torch.min(ideal_gradient)

            dot_product = torch.dot(ideal_gradient.view(-1), achieved_gradient.view(-1))

            loss = dot_product + CA_c * torch.norm(ideal_gradient) * torch.norm(achieved_gradient)

            if loss.item() < loss_best:
                loss_best = loss.item()
                U_best = U.detach().clone()
                V_best = V.detach().clone()
            if step < CA_step:
                loss.backward()
                optimizer.step()
                scheduler.step()

            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item()}")

        return None, None

    def extract_lora_AB_gradients(self, models):
        # Get global model state dict
        global_state_dict = self.model.state_dict()
        
        # Extract global LoRA parameters in sorted order
        global_A_dict = {}
        global_B_dict = {}
        
        for name, param in global_state_dict.items():
            if "lora_A" in name:
                global_A_dict[name] = param.detach().clone()
            elif "lora_B" in name:
                global_B_dict[name] = param.detach().clone()
        
        # Sort the names to ensure consistent ordering
        sorted_A_names = sorted(global_A_dict.keys())
        sorted_B_names = sorted(global_B_dict.keys())
        
        grad_A_all = []
        grad_B_all = []
        
        for i in range(len(models)):
            _, client_state_dict = models[i]
            
            # Extract client LoRA parameters in the same sorted order
            client_A_dict = {}
            client_B_dict = {}
            
            for name, param in client_state_dict.items():
                if "lora_A" in name:
                    client_A_dict[name] = param.detach().clone()
                elif "lora_B" in name:
                    client_B_dict[name] = param.detach().clone()
            
            # Compute gradients (client - global) in sorted order
            grad_A_list = []
            grad_B_list = []
            
            for name in sorted_A_names:
                if name in client_A_dict:
                    grad = client_A_dict[name] - global_A_dict[name]
                    grad_A_list.append(grad)

            for name in sorted_B_names:
                if name in client_B_dict:
                    grad = client_B_dict[name] - global_B_dict[name]
                    grad_B_list.append(grad)

            grad_A_all.append(torch.stack(grad_A_list))  # shape: [num_layers, ...]
            grad_B_all.append(torch.stack(grad_B_list))
        
        grad_A_all = torch.stack(grad_A_all)  # shape: [num_clients, num_layers, ...]
        grad_B_all = torch.stack(grad_B_all)
        
        return grad_A_all, grad_B_all

    def extract_lora_AB(self, models):
        A_all = []
        B_all = []

        for i in range(len(models)):
            _, state_dict = models[i]

            # Use dictionaries to store parameters with their names
            A_dict = {}
            B_dict = {}

            for name, param in state_dict.items():
                if "lora_A" in name:
                    A_dict[name] = param.detach().clone()
                elif "lora_B" in name:
                    B_dict[name] = param.detach().clone()

            # Sort the names to ensure consistent ordering
            sorted_A_names = sorted(A_dict.keys())
            sorted_B_names = sorted(B_dict.keys())
            
            # Create lists in sorted order
            A_list = [A_dict[name] for name in sorted_A_names]
            B_list = [B_dict[name] for name in sorted_B_names]

            A_all.append(torch.stack(A_list))  # shape: [num_layers, ...]
            B_all.append(torch.stack(B_list))

        A_all = torch.stack(A_all)  # shape: [num_clients, num_layers, ...]
        B_all = torch.stack(B_all)
        
        return A_all, B_all

    def compute_pairwise_mse(self, A_all, B_all):
        num_clients = A_all.shape[0]
        pairwise_mse = {}

        for i, j in itertools.combinations(range(num_clients), 2):

            mse_A = F.mse_loss(A_all[i], A_all[j])
            mse_B = F.mse_loss(B_all[i], B_all[j])
            
            total_mse = mse_A + mse_B

            pairwise_mse[(i, j)] = {
                'mse_A': mse_A.item(),
                'mse_B': mse_B.item(),
                'total': total_mse.item()
            }

        return pairwise_mse

        
class OnlineExactClientsAggregator(ExactClientsAggregator):
    """
    Implementation of online aggregation of FedAvg.
    """
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineExactClientsAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        """
        Reset the state of the model to its initial state
        """
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        """
        Increment the model weight by the given content.
        """
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        """
        Returns the aggregated value
        """
        return self.maintained