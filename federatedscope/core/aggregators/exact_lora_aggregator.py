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

        # TODO 2. Split grad_A and grad_B
        # print(f"model keys: {models[0][1].keys()}")
        """
        lora_A_list = []
        lora_B_list = []
        for name, param in model.named_parameters():
            if "lora_A" in name:
                lora_A_list.append(SOMETHING)
            if "lora_B" in name:
                lora_B_list.append(SOMETHING)    
            flat the LIST
            
        Tensor -> List in Line 196 - 202 of the following link:
        Gradient -> Tensor in Line 232 - 233 of the following link:
        https://github.com/chickbong221/FCL/blob/PFLlib-based/system/flcore/servers/serverstgm.py
        """

        # TODO 2'. self._para_weighted_avg grad_A and grad_B, and assign back to avg_model_2
        # FIXME The results of avg_model_2 must be equal to avg_model
        # TODO 3. NOTYET: Find Grad_A and Grad_B according to equation in paper

        """
        meta_weights = self.stgm_high(
                        meta_weights=self.global_model,
                        inner_weights=self.uploaded_models,
                        lr_meta= self.stgm_meta_lr
                    )
        self.global_model.load_state_dict(copy.deepcopy(meta_weights))
        
        - And we will change the function in Line 265 and Line 322-327
        """
        # avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)
        # avg_model_2 = self._lora_weighted_avg(models, recover_fun=recover_fun)

        # # Test if avg_model and avg_model_2 are approximately equal
        # for key in avg_model:
        #     if key in avg_model_2:
        #         tensor1 = avg_model[key].detach()
        #         tensor2 = avg_model_2[key].detach()
        #         if not torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-3):
        #             raise ValueError(
        #                 f"Mismatch in aggregated parameters for {key}:\n"
        #                 f"Max diff: {(tensor1 - tensor2).abs().max()}\n"
        #                 f"{tensor1} vs {tensor2}"
        #             )

        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
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

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Apply Exact LoRA scaling using U, V for each client,
        then calculate the weighted average of models.
        """
        A_all, B_all = self.extract_lora_AB(models)

        A_all_copy = A_all.clone()
        B_all_copy = B_all.clone()

        value = A_all_copy[0].norm(p=2).item()
        print(f"{value:.10f}")


        # print(f"Shape of A_all: {A_all.shape}")

        # print(A_all_copy.norm(p=2))

        diffs = self.compute_pairwise_mse(A_all_copy, B_all_copy)

        # for (i, j), vals in diffs.items():
        #     print(f"Client pair ({i}, {j}): MSE_A = {vals['mse_A']}, MSE_B = {vals['mse_B']}, Total = {vals['total']}")
        
        U, V = self.optimize_uv(A_all, B_all, lr=5e-3, steps=100)

        num_clients = len(models)
        training_set_size = sum(sample_size for sample_size, _ in models)

        sample_size, avg_model = models[0]

        for key in avg_model:
            for i in range(num_clients):
                local_sample_size, local_model = models[i]

                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / num_clients
                elif self.cfg.federate.use_ss:
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])

                # Scale with U[i] or V[i] if key contains lora_A or lora_B
                scaled_tensor = local_model[key]
                # print(U[i].device, V[i].device, scaled_tensor.device)
                if "lora_A" in key:
                    scaled_tensor = U[i] * scaled_tensor
                elif "lora_B" in key:
                    scaled_tensor = V[i] * scaled_tensor

                if i == 0:
                    avg_model[key] = scaled_tensor * weight
                else:
                    avg_model[key] += scaled_tensor * weight

            # Secret sharing post-processing
            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model

    def optimize_uv(self, A_all, B_all, lr=5e-3, steps=50):

        A_all = A_all.to(self.device)
        B_all = B_all.to(self.device)

        num_clients = A_all.shape[0]
        BA = []

        for i in range(len(A_all)):
            BA.append(torch.matmul(B_all[i], A_all[i]))
        ideal_update = torch.stack(BA).mean(dim=0)
        # print(ideal_update)

        U = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        V = torch.nn.Parameter(torch.ones(num_clients, device=self.device))

        optimizer = torch.optim.SGD([U, V], lr=lr)

        # for step in range(steps):
        #     optimizer.zero_grad()

        #     UA = (U.view(-1, 1, 1, 1) * A_all).mean(dim=0)  # [num_layers, ...]
        #     VB = (V.view(-1, 1, 1, 1) * B_all).mean(dim=0)
        #     naive_update = torch.matmul(VB, UA)  # [num_layers, ...]

        #     # print(naive_update)
        #     # loss_fn = torch.nn.L1Loss()
        #     # loss = loss_fn(naive_update, ideal_update)
        #     loss = torch.mean((naive_update - ideal_update) ** 2) 
            
        #     loss.backward()
        #     optimizer.step()

        #     if step % 10 == 0:
        #         print(f"Step {step}: Loss = {loss.item()}")

        # UA = (U.view(-1, 1, 1, 1) * A_all).mean(dim=0)  # [num_layers, ...]
        # VB = (V.view(-1, 1, 1, 1) * B_all).mean(dim=0)
        # naive_update = torch.matmul(VB, UA)  # [num_layers, ...]

        # # loss_fn = torch.nn.MSELoss()
        # # loss_fn = torch.nn.L1Loss()
        # # loss = loss_fn(naive_update, ideal_update)
        # loss = torch.mean((naive_update - ideal_update) ** 2)

        # print(f"Loss = {loss.item()}")

        return U.detach().squeeze().cpu(), V.detach().squeeze().cpu()  # shape: [num_clients]


    def extract_lora_AB(self, models):
        A_all = []
        B_all = []

        for i in range(0, len(models)):
            _, state_dict = models[i]

            A_list = []
            B_list = []

            for name, param in state_dict.items():
                if "lora_A" in name:
                    # print(f"Extracting {name} with shape {param}")
                    A_list.append(param.detach().clone())
                elif "lora_B" in name:
                    B_list.append(param.detach().clone())

            A_list = sorted(A_list, key=lambda x: x.shape)
            B_list = sorted(B_list, key=lambda x: x.shape)

            A_all.append(torch.stack(A_list))  # shape: [num_layers, ...]
            B_all.append(torch.stack(B_list))

        A_all = torch.stack(A_all)  # shape: [num_clients, num_layers, ...]
        B_all = torch.stack(B_all)

        # norm_A = A_all.norm(p=2)  # scalar
        # norm_B = B_all.norm(p=2)

        # print(f"Norm of A_all: {norm_A.item()}, Norm of B_all: {norm_B.item()}")
        # print(f"Shape of A_all: {A_all.shape}, Shape of B_all: {B_all.shape}")
        # print(f"Max-Min of A_all: {A_all.min()}, {A_all.max()}")
        # print(f"Max-Min of B_all: {B_all.min()}, {B_all.max()}")

        def normalize_minmax(tensor):
            min_val = tensor.min()
            max_val = tensor.max()
            return (tensor - min_val) / (max_val - min_val + 1e-8)

        def normalize_l2(tensor):
            norm = tensor.norm(p=2)
            return tensor / (norm + 1e-8)

        # A_all = normalize_minmax(A_all)
        # B_all = normalize_minmax(B_all)

        # A_all = normalize_l2(A_all)
        # B_all = normalize_l2(B_all)

        # norm_A = A_all.norm(p=2)  # scalar
        # norm_B = B_all.norm(p=2)

        # print(f"Norm of A_all after nor: {norm_A.item()}, Norm of B_all after nor: {norm_B.item()}")
        # print(f"Shape of A_all after nor: {A_all.shape}, Shape of B_all after nor: {B_all.shape}")

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
