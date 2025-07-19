import os
import torch
import torch.nn as nn
import torch.optim as optim
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor

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
        print(f"Model num: {len(models)}")
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
        U, V = self.optimize_uv(A_all, B_all, lr=1e-2, steps=50)

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


    # Test if exacting LoRA parameters is correct
    # def _lora_weighted_avg(self, models, recover_fun=None):
    #     total_size = sum(sample_size for sample_size, _ in models)
    #     _, reference_model = models[0]

    #     avg_lora = {}

    #     for key in reference_model:
    #         if "lora_A" not in key and "lora_B" not in key:
    #             continue  # skip non-LoRA parameters

    #         for i, (sample_size, local_model) in enumerate(models):
    #             # Decide weight
    #             if self.cfg.federate.ignore_weight:
    #                 weight = 1.0 / len(models)
    #             elif self.cfg.federate.use_ss:
    #                 weight = 1.0  # assume client already multiplies by sample_size
    #             else:
    #                 weight = sample_size / total_size

    #             param = local_model[key]
    #             if not self.cfg.federate.use_ss:
    #                 param = param2tensor(param)

    #             if i == 0:
    #                 avg_lora[key] = param * weight
    #             else:
    #                 avg_lora[key] += param * weight

    #         # If using secret sharing
    #         if self.cfg.federate.use_ss and recover_fun:
    #             avg_lora[key] = recover_fun(avg_lora[key])
    #             avg_lora[key] /= total_size
    #             avg_lora[key] = torch.FloatTensor(avg_lora[key])

    #     return avg_lora

    def optimize_uv(self, A_all, B_all, lr=1e-2, steps=50):
        num_clients = A_all.shape[0]

        U = torch.nn.Parameter(torch.ones(num_clients, device=self.device))
        V = torch.nn.Parameter(torch.ones(num_clients, device=self.device))

        optimizer = torch.optim.SGD([U, V], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()

            UA = (U.view(-1, 1, 1, 1) * A_all).mean(dim=0)  # [num_layers, ...]
            VB = (V.view(-1, 1, 1, 1) * B_all).mean(dim=0)
            
            BA = (torch.matmul(B_all, A_all)).mean(dim=0)

            loss = (torch.matmul(VB, UA).sum() - BA.sum()) ** 2
            print(torch.matmul(VB, UA).sum(), BA.sum())
            
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.6f}")

        return U.squeeze(), V.squeeze()  # shape: [num_clients]


    def extract_lora_AB(self, models):
        A_all = []
        B_all = []

        for i in range(0, len(models)):
            _, state_dict = models[i]

            A_list = []
            B_list = []
            for name, param in state_dict.items():
                if "lora_A" in name:
                    A_list.append(param.detach().clone())
                elif "lora_B" in name:
                    B_list.append(param.detach().clone())

            A_list = sorted(A_list, key=lambda x: x.shape)
            B_list = sorted(B_list, key=lambda x: x.shape)

            A_all.append(torch.stack(A_list))  # shape: [num_layers, ...]
            B_all.append(torch.stack(B_list))

        A_all = torch.stack(A_all)  # shape: [num_clients, num_layers, ...]
        B_all = torch.stack(B_all)

        return A_all, B_all


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
