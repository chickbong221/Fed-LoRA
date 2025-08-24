import os
import torch
import torch.nn as nn
import torch.optim as optim
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import torch.nn.functional as F
import itertools

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
        Apply Exact and Conflict-free LoRA scaling using optimized U, V for gradients,
        then calculate the weighted average and update the global model.
        """
        # Get optimized gradients from the optimization process
        exact_grad_A, exact_grad_B, sorted_A_names, sorted_B_names = \
            self.optimize_exact_gradient(models, lr=1e-2, steps=150)

        num_clients = len(models)
        training_set_size = sum(sample_size for sample_size, _ in models)
        
        global_state_dict = self.model.state_dict()
        avg_model = {key: param.clone() for key, param in global_state_dict.items()}
        
        # Build dictionary of optimized gradients {param_name: [per-client gradients]}
        optimized_gradients = {}
        for i, name in enumerate(sorted_A_names):
            optimized_gradients[name] = [exact_grad_A[c, i] for c in range(num_clients)]
        for i, name in enumerate(sorted_B_names):
            optimized_gradients[name] = [exact_grad_B[c, i] for c in range(num_clients)]
        
        # Apply weighted average
        for key in global_state_dict:
            total_weighted_gradient = torch.zeros_like(global_state_dict[key])
            
            if key in optimized_gradients:
                for i in range(num_clients):
                    local_sample_size, _ = models[i]
                    if self.cfg.federate.ignore_weight:
                        weight = 1.0 / num_clients
                    elif self.cfg.federate.use_ss:
                        weight = 1.0
                    else:
                        weight = local_sample_size / training_set_size
                    
                    gradient = optimized_gradients[key][i].detach().clone().cpu()
                    total_weighted_gradient += gradient * weight
            
            # Secret sharing post-processing
            if self.cfg.federate.use_ss and recover_fun:
                total_weighted_gradient = recover_fun(total_weighted_gradient)
                total_weighted_gradient /= training_set_size
                total_weighted_gradient = torch.FloatTensor(total_weighted_gradient)
            
            avg_model[key] = global_state_dict[key] + total_weighted_gradient
        
        return avg_model

    def optimize_exact_gradient(self, models, lr=5e-3, steps=50):
        """
        Optimize U, V parameters and return the scaled gradients that achieve
        the best approximation to the ideal LoRA update.
        """
        # Extract LoRA parameters and gradients
        A_all, B_all = self.extract_lora_AB(models)
        grad_A_all, grad_B_all, sorted_A_names, sorted_B_names = self.extract_lora_AB_gradients(models)
        
        # Move to device
        device_tensors = [A_all, B_all, grad_A_all, grad_B_all]
        A_all, B_all, grad_A_all, grad_B_all = [t.to(self.device) for t in device_tensors]
        
        num_clients = A_all.shape[0]
        
        # Get global LoRA parameters
        global_A, global_B, _, _ = self._get_global_lora_params()
        
        # Compute ideal update (target)
        ideal_update = torch.stack([
            torch.matmul(B_all[i], A_all[i]) for i in range(num_clients)
        ]).mean(dim=0)
        
        # Initialize optimization parameters
        U = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        V = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        
        optimizer = torch.optim.AdamW([U, V], lr=lr)
        
        best_loss = float('inf')
        best_U, best_V = None, None
        
        # Optimization loop
        for step in range(steps):
            optimizer.zero_grad()
            
            # Apply U, V scaling to gradients
            scaled_grad_A = (U.view(-1, 1, 1, 1) * grad_A_all).mean(dim=0)
            scaled_grad_B = (V.view(-1, 1, 1, 1) * grad_B_all).mean(dim=0)
            
            # Compute achieved update
            updated_A = global_A + scaled_grad_A
            updated_B = global_B + scaled_grad_B
            achieved_update = torch.matmul(updated_B, updated_A)
            
            # Loss: normalized MSE
            diff = achieved_update - ideal_update
            normalized_diff = diff / (torch.abs(diff).max() + 1e-8)
            loss = torch.mean(normalized_diff ** 2)
            
            # Track best parameters
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_U = U.detach().clone()
                best_V = V.detach().clone()
            
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}: Loss = {loss.item()}")

        print(f"Best loss achieved: {best_loss}")
        
        # Compute final scaled gradients (per client, not averaged)
        scaled_grad_A = (best_U.view(-1, 1, 1, 1) * grad_A_all)
        scaled_grad_B = (best_V.view(-1, 1, 1, 1) * grad_B_all)

        # Return both the gradients and their parameter name order
        return scaled_grad_A, scaled_grad_B, sorted_A_names, sorted_B_names

    def optimize_conflict_free_gradient(self, models, lr_meta):
        
        # Extract current parameters and gradients
        grad_A_all, grad_B_all = self.extract_lora_AB_gradients(models)
        
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
        optimizer = torch.optim.SGD([U, V], lr=self.stgm_learning_rate, momentum=self.stgm_momentum)
        scheduler = StepLR(optimizer, step_size=self.stgm_step_size, gamma=self.stgm_gamma)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Apply weighted gradients to global parameters
            weighted_grad_A = (U.view(-1, 1, 1, 1) * grad_A_all).mean(dim=0)  # [num_layers, ...]
            weighted_grad_B = (V.view(-1, 1, 1, 1) * grad_B_all).mean(dim=0)  # [num_layers, ...]

            achieved_gradient = torch.matmul(weighted_grad_B, weighted_grad_A)

            # Loss: difference between achieved update and ideal update
            ideal_gradient = ideal_gradient.float() 
            achieved_gradient = achieved_gradient.float()

    def optimize_conflict_free_gradient(self, models, lr_meta):
        # Extract LoRA gradients
        grad_A_all, grad_B_all = self.extract_lora_AB_gradients(models)
        
        all_A_grads = []
        all_B_grads = []
        
        for i_client in range(len(models)):
            # Get A and B gradients for this client
            grad_A_client = grad_A_all[i_client]  # [num_layers, ...]
            grad_B_client = grad_B_all[i_client]  # [num_layers, ...]
            
            A_grads_flat = []
            B_grads_flat = []
            
            for layer_idx in range(grad_A_client.shape[0]):
                # Also store flattened A and B gradients separately
                A_grads_flat.append(grad_A_client[layer_idx].flatten())
                B_grads_flat.append(grad_B_client[layer_idx].flatten())
            
            # Concatenate gradients
            A_grad_vector = torch.cat(A_grads_flat)  # LoRA A size
            B_grad_vector = torch.cat(B_grads_flat)  # LoRA B size
            
            all_A_grads.append(A_grad_vector)
            all_B_grads.append(B_grad_vector)
        
        # Gradient normalization
        if self.grad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            A_grad_norms = [torch.norm(grad) for grad in all_A_grads]
            B_grad_norms = [torch.norm(grad) for grad in all_B_grads]
            
            # Step 2: Determine scaling factors to balance the norms
            target_norm_A = torch.mean(torch.tensor(A_grad_norms))
            target_norm_B = torch.mean(torch.tensor(B_grad_norms))
            
            scaling_factors_A = [target_norm_A / norm if norm > 0 else 1.0 for norm in A_grad_norms]
            scaling_factors_B = [target_norm_B / norm if norm > 0 else 1.0 for norm in B_grad_norms]
            
            # Step 3: Scale gradient vectors
            balanced_A_grads = [grad * scale for grad, scale in zip(all_A_grads, scaling_factors_A)]
            balanced_B_grads = [grad * scale for grad, scale in zip(all_B_grads, scaling_factors_B)]
            
            # Step 4: Stack the balanced gradients into tensors
            all_A_grads_tensor = torch.stack(balanced_A_grads).t()
            all_B_grads_tensor = torch.stack(balanced_B_grads).t()
        else:
            all_A_grads_tensor = torch.stack(all_A_grads).t()
            all_B_grads_tensor = torch.stack(all_B_grads).t()
        
        # Apply Conflict-free Optimization with separate U and V optimization
        g_effective = self.optimize_conflict_free_core(all_domains_grad_tensor, all_A_grads_tensor, all_B_grads_tensor, len(models), grad_A_all, grad_B_all)
        
        return g_effective

    def optimize_conflict_free_core(self, grad_effective_vec, grad_A_vec, grad_B_vec, num_clients, grad_A_all, grad_B_all):

        grads_A = grad_A_vec.to(self.device)
        grads_B = grad_B_vec.to(self.device)
        
        # Initialize U and V weights
        U = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        V = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        
        # Setup optimizer for both U and V
        optimizer = torch.optim.SGD([U, V], lr=self.stgm_learning_rate, momentum=self.stgm_momentum)
        scheduler = StepLR(optimizer, step_size=self.stgm_step_size, gamma=self.stgm_gamma)
        
        U_best = None
        V_best = None
        obj_best = np.inf
        
        for i in range(self.stgm_rounds + 1):
            optimizer.zero_grad()
            
            # Apply softmax to get normalized weights
            uu = torch.softmax(U, dim=0)  # weights for A gradients
            vv = torch.softmax(V, dim=0)  # weights for B gradients
            
            # Compute weighted effective gradients by reconstructing V*B @ U*A
            weighted_effective_grads = []
            
            for client_idx in range(num_clients):
                # Get individual client gradients
                client_weighted_effective_grad = []
                
                # Reconstruct weighted effective gradient for this client
                grad_idx_A = 0
                grad_idx_B = 0
                
                for layer_idx in range(grad_A_all.shape[1]):  # num_layers
                    # Get shapes for this layer
                    A_layer_shape = grad_A_all[client_idx, layer_idx].shape  # [r, in_dim]
                    B_layer_shape = grad_B_all[client_idx, layer_idx].shape  # [out_dim, r]
                    
                    A_layer_size = A_layer_shape[0] * A_layer_shape[1]
                    B_layer_size = B_layer_shape[0] * B_layer_shape[1]
                    
                    # Extract layer gradients
                    A_layer_grad = grads_A[grad_idx_A:grad_idx_A + A_layer_size, client_idx].reshape(A_layer_shape)
                    B_layer_grad = grads_B[grad_idx_B:grad_idx_B + B_layer_size, client_idx].reshape(B_layer_shape)
                    
                    # Apply weights and compute effective gradient: V*B @ U*A
                    weighted_A = A_layer_grad * uu[client_idx]
                    weighted_B = B_layer_grad * vv[client_idx]
                    layer_effective_grad = torch.matmul(weighted_B, weighted_A)
                    
                    client_weighted_effective_grad.append(layer_effective_grad.flatten())
                    
                    grad_idx_A += A_layer_size
                    grad_idx_B += B_layer_size
                
                # Concatenate all layers for this client
                client_effective_vector = torch.cat(client_weighted_effective_grad)
                weighted_effective_grads.append(client_effective_vector)
            
            # Stack weighted effective gradients
            combined_grads = torch.stack(weighted_effective_grads).t()  # [grad_dim_full, num_clients]
            
            # Compute GG matrix for the combined effective gradients
            GG = combined_grads.t().mm(combined_grads)  # [num_clients, num_clients]
            
            # Normalization
            scale = (torch.diag(GG) + 1e-4).sqrt().mean()
            GG = GG / scale.pow(2)
            Gg = GG.mean(1, keepdims=True)  # [num_clients, 1]
            gg = Gg.mean(0, keepdims=True)  # [1, 1]
            
            # Regularization parameter
            c = (gg + 1e-4).sqrt() * self.stgm_c
            
            # Compute combined weights for objective (average of U and V weights)
            ww_combined = (uu + vv) / 2  # [num_clients]
            
            # Objective function using combined weights
            obj = ww_combined.t().mm(Gg) + c * (ww_combined.t().mm(GG).mm(ww_combined.unsqueeze(1)) + 1e-4).sqrt()
            
            if obj.item() < obj_best:
                obj_best = obj.item()
                U_best = U.clone()
                V_best = V.clone()
            
            if i < self.stgm_rounds:
                obj.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
    
        # Use best weights found to compute final effective gradient
        uu_best = torch.softmax(U_best, dim=0)
        vv_best = torch.softmax(V_best, dim=0)
        
        # Compute final weighted effective gradients
        final_weighted_effective_grads = []
        
        for client_idx in range(num_clients):
            client_weighted_effective_grad = []
            grad_idx_A = 0
            grad_idx_B = 0
            
            for layer_idx in range(grad_A_all.shape[1]):  # num_layers
                A_layer_shape = grad_A_all[client_idx, layer_idx].shape
                B_layer_shape = grad_B_all[client_idx, layer_idx].shape
                
                A_layer_size = A_layer_shape[0] * A_layer_shape[1]
                B_layer_size = B_layer_shape[0] * B_layer_shape[1]
                
                A_layer_grad = grads_A[grad_idx_A:grad_idx_A + A_layer_size, client_idx].reshape(A_layer_shape)
                B_layer_grad = grads_B[grad_idx_B:grad_idx_B + B_layer_size, client_idx].reshape(B_layer_shape)
                
                # Apply best weights
                weighted_A = A_layer_grad * uu_best[client_idx]
                weighted_B = B_layer_grad * vv_best[client_idx]
                layer_effective_grad = torch.matmul(weighted_B, weighted_A)
                
                client_weighted_effective_grad.append(layer_effective_grad.flatten())
                
                grad_idx_A += A_layer_size
                grad_idx_B += B_layer_size
            
            client_effective_vector = torch.cat(client_weighted_effective_grad)
            final_weighted_effective_grads.append(client_effective_vector)
        
        final_combined_grads = torch.stack(final_weighted_effective_grads).t()
        
        # Compute final aggregated gradient using STGM formula
        GG_final = final_combined_grads.t().mm(final_combined_grads)
        scale_final = (torch.diag(GG_final) + 1e-4).sqrt().mean()
        GG_final = GG_final / scale_final.pow(2)
        
        ww_avg = (uu_best + vv_best) / 2
        gw_norm = (ww_avg.t().mm(GG_final).mm(ww_avg.unsqueeze(1)) + 1e-4).sqrt()
        c_final = (GG_final.mean() + 1e-4).sqrt() * self.stgm_c
        lmbda = c_final / (gw_norm + 1e-4)
        
        g_effective = ((1 / num_clients + ww_avg * lmbda).view(-1, 1) * final_combined_grads.t()).sum(0) / (1 + self.stgm_c ** 2)
        
        return g_effective


    def extract_lora_AB_gradients(self, models):
        """
        Extract LoRA A and B gradients for all clients in a clean, consistent format.
        """
        global_state_dict = self.model.state_dict()
        
        # Get sorted LoRA parameter names for consistency
        sorted_A_names = sorted([name for name in global_state_dict if "lora_A" in name])
        sorted_B_names = sorted([name for name in global_state_dict if "lora_B" in name])
        
        grad_A_all, grad_B_all = [], []
        
        for _, client_state_dict in models:
            grad_A_list, grad_B_list = [], []
            
            for name in sorted_A_names:
                if name in client_state_dict:
                    client_param = client_state_dict[name].detach().clone()
                    global_param = global_state_dict[name].detach().clone()
                    gradient = client_param - global_param
                    grad_A_list.append(gradient)
            
            for name in sorted_B_names:
                if name in client_state_dict:
                    client_param = client_state_dict[name].detach().clone()
                    global_param = global_state_dict[name].detach().clone()
                    gradient = client_param - global_param
                    grad_B_list.append(gradient)
            
            if grad_A_list:
                grad_A_all.append(torch.stack(grad_A_list))
            if grad_B_list:
                grad_B_all.append(torch.stack(grad_B_list))
        
        return torch.stack(grad_A_all), torch.stack(grad_B_all), sorted_A_names, sorted_B_names

    def extract_lora_AB(self, models):
        """
        Extract LoRA A and B parameters for all clients in a consistent format.
        """
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

    def _get_global_lora_params(self):
        """
        Helper method to extract global LoRA parameters in sorted order.
        """
        global_state_dict = self.model.state_dict()
        
        # Extract and sort LoRA parameter names
        global_A_dict = {name: param.detach().clone() for name, param in global_state_dict.items() if "lora_A" in name}
        global_B_dict = {name: param.detach().clone() for name, param in global_state_dict.items() if "lora_B" in name}
        
        sorted_A_names = sorted(global_A_dict.keys())
        sorted_B_names = sorted(global_B_dict.keys())
        
        # Stack parameters in consistent order
        global_A = torch.stack([global_A_dict[name] for name in sorted_A_names]).to(self.device)
        global_B = torch.stack([global_B_dict[name] for name in sorted_B_names]).to(self.device)
        
        return global_A, global_B, sorted_A_names, sorted_B_names

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
