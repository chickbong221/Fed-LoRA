import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import itertools
import torch.nn.functional as F


def enable_adapter(model, package, adapter, **kwargs):
    """
    Enables an adapter for a given model and package.

    Args:
        model: A pre-trained model from HuggingFace Transformers library.
        package: A string indicating the name of the package that provides
            the adapter. Currently, only 'peft' and 'adapterhub' is supported.
        adapter: A string indicating the name of the adapter to enable. The
            available adapters depend on the package.
        **kwargs: Additional keyword arguments that are passed to the
            adapter configuration.

    Returns:
        A model object that has the adapter enabled.

    Raises:
        NotImplementedError: If the package or the adapter is not supported.
    """
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """

        A_all = []
        B_all = []

        def compute_pairwise_mse(A_all, B_all):
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

        def extract_lora_matrices(model):
            state_dict = model.state_dict()

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

            return A_list, B_list

        def reset_lora_A_B_random(model):
            import torch.nn as nn

            count = 0
            for _, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    if isinstance(module.lora_A, nn.ModuleDict):
                        for sub in module.lora_A.values():
                            if hasattr(sub, "weight"):
                                nn.init.normal_(sub.weight, mean=0.0, std=0.02)
                                count += 1
                        for sub in module.lora_B.values():
                            if hasattr(sub, "weight"):
                                nn.init.normal_(sub.weight, mean=0.0, std=0.02)
                    else:
                        if hasattr(module.lora_A, "weight"):
                            nn.init.normal_(module.lora_A.weight, mean=0.0, std=0.02)
                            count += 1
                        if hasattr(module.lora_B, "weight"):
                            nn.init.normal_(module.lora_B.weight, mean=0.0, std=0.02)
            
        from peft import get_peft_model, TaskType
        if adapter == 'lora':
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, **kwargs)
            model = get_peft_model(model, peft_config)
            reset_lora_A_B_random(model)

            # model_toy1 = copy.deepcopy(model)
            # model_toy2 = copy.deepcopy(model)

            # model_toy1 = get_peft_model(model_toy1, peft_config)
            # model_toy2 = get_peft_model(model_toy2, peft_config)

            # A_list, B_list = extract_lora_matrices(model_toy1)
            # A_all.append(torch.stack(A_list))
            # B_all.append(torch.stack(B_list))

            # A_list, B_list = extract_lora_matrices(model_toy2)
            # A_all.append(torch.stack(A_list))
            # B_all.append(torch.stack(B_list))

            # A_all = torch.stack(A_all)
            # B_all = torch.stack(B_all)

            # diffs = compute_pairwise_mse(A_all, B_all)

            # for (i, j), vals in diffs.items():
            #     print(f"Client pair ({i}, {j}): MSE_A = {vals['mse_A']}, MSE_B = {vals['mse_B']}, Total = {vals['total']}")


            # A_list = []
            # B_list = []

            # for name, param in model.named_parameters():
            #     if "lora_a" in name.lower():
            #         A_list.append(param.detach().flatten())
            #     elif "lora_b" in name.lower():
            #         B_list.append(param.detach().flatten())

            # if A_list:
            #     A_tensor = torch.cat(A_list)
            #     print(f"Total LoRA A norm: {A_tensor.norm(p=2)}")

            # if B_list:
            #     B_tensor = torch.cat(B_list)
            #     print(f"Total LoRA B norm: {B_tensor.norm(p=2)}")


        # added by me, for VeRA
        elif adapter == 'vera':
            from peft import VeraConfig
            peft_config = VeraConfig(task_type=TaskType.SEQ_CLS, **kwargs)
            model = get_peft_model(model, peft_config)
        # added by me, for rsLoRA
        elif adapter == 'rslora':
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prefix':
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_CLS,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        model.print_trainable_parameters()

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        # TODO:  After supporting adapterhub, we will move the following
        #   parameters in yaml file for users' convenient
        if adapter == 'lora':
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            from transformers.adapters import AdapterConfig, ConfigUnion

            # TODO: configure these args in cfg
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
    else:
        raise NotImplementedError
    return model


class AdapterModel(nn.Module):
    """
    A wrapper class for a model that can use adapters for fine-tuning.

    This class inherits from torch.nn.Module and implements a wrapper for a
    model that can optionally use adapters for fine-tuning. Adapters are small
    modules that can be inserted between the layers of a pretrained model and
    trained on a specific task, while keeping the original parameters frozen.
    This class can use different adapter packages and methods, such as PEFT
    and LoRA. It also provides methods for saving and loading the model state
    dict, as well as generating text using the model.

    Attributes:
        model: A torch.nn.Module object that represents the original or
            adapted model.

    """
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        """
        Initializes the wrapper with the given model and arguments.

        Args:
            model: A torch.nn.Module object that represents the original model.
            use_adapter: A boolean indicating whether to use adapters for
                fine-tuning. Default is False.
            *args: Additional positional arguments to pass to the adapter
                package or method.
            **kwargs: Additional keyword arguments to pass to the adapter
                package or method. These may include adapter_package,
                adapter_method, etc.
        """
        super().__init__()

        self.model = None
        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            self.model = enable_adapter(model, adapter_package, adapter_method,
                                        **kwargs)
        else:
            self.model = model

    def forward(self, *args, **kwargs):
        """
        Calls the forward method of the wrapped model.

        Args:
            *args: Positional arguments to pass to the model's forward method.
            **kwargs: Keyword arguments to pass to the model's forward method.

        Returns:
            The output of the model's forward method.
        """
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Calls the generate method of the wrapped model.

        Args:
            *args: Positional arguments to pass to the model's generate method.
            **kwargs: Keyword arguments to pass to the model's generate method.

        Returns:
            The output of the model's generate method.
        """
        try:
            res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # When does evaluation in HELM,
            # half precision will cause RuntimeError,
            # the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        return res

    def state_dict(self, return_trainable=True, *args, **kwargs):
        """
        Returns the state dict of the wrapped model.

        Args:
            return_trainable: A boolean indicating whether to return only the
                trainable parameters of the model. Default is True.
            *args: Additional positional arguments to pass to the model's
                state_dict method.
            **kwargs: Additional keyword arguments to pass to the model's
                state_dict method.

        Returns:
            A dictionary containing the state dict of the model. If
            return_trainable is True, only the parameters that require grad are
            included. Otherwise, all parameters are included.
        """
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        """
        Loads the state dict into the wrapped model.

        Args:
            state_dict: A dictionary containing the state dict to load into
                the model.
            strict: A boolean indicating whether to strictly enforce that the
                keys in state_dict match the keys returned by this module’s
                state_dict() function. Default is False.
        """
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        """
        Returns only the trainable parameters of the wrapped model.

        This method can be used to get only the parameters that require grad,
        such as adapters or task-specific layers.

        Returns:
            A dictionary containing the state dict of the trainable parameters
            of the model.
        """
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self, path, state=0):
        """
        Saves the model state dict and the current round to a file.

        Args:
            path: A string representing the file path to save the model to.
            state: An integer representing the current round of training or
                evaluation. Default is 0.

        """
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    # TODO: Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)
