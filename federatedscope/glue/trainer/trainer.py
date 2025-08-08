import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.glue.model.adapter_builder import AdapterModel
from datasets import load_metric
import numpy as np

logger = logging.getLogger(__name__)


class GLUETrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                if ctx.cfg.llm.adapter.args[0].get('adapter_method', '') == "vera":
                    # added by me, for VeRA, introduce separate learning rates for the classification head and the adapted layers
                    vera_params = [param for name, param in ctx.model.named_parameters() if "vera" in name and param.requires_grad]
                    other_params = [param for name, param in ctx.model.named_parameters() if "vera" not in name and param.requires_grad]
                    optimizer_grouped_parameters = [
                        {'params': vera_params, 'lr': ctx.cfg.train.optimizer.lr},
                        {'params': other_params, 'lr': ctx.cfg.train.vera.lr_c}
                    ]
                    from transformers import AdamW, get_linear_schedule_with_warmup
                    ctx.optimizer = AdamW(optimizer_grouped_parameters, no_deprecation_warning=True)
                    ctx.scheduler = get_linear_schedule_with_warmup(
                                    ctx.optimizer, 
                                    num_warmup_steps=0.06 * ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num, 
                                    num_training_steps=ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num
                    )
                else:
                    ctx.optimizer = get_optimizer(
                        ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                    ctx.scheduler = get_scheduler(
                        ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)    # modified by me, for GLUE

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        
        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        preds = outputs.logits.argmax(dim=-1)  # modified by me, for GLUE
        loss = outputs.loss
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(preds, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        def get_lora_matrices_norm(model):
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

            norm_A, norm_B = None, None

            if A_list:
                A_all = torch.cat(A_list)
                norm_A = A_all.norm(2).item()

            if B_list:
                B_all = torch.cat(B_list)
                norm_B = B_all.norm(2).item()

            return norm_A, norm_B
        
        def get_lora_AB_grad_norm(model, norm_type=2):
            grad_list_A = []
            grad_list_B = []

            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                if "lora_A" in name:
                    grad = param.grad.detach()
                    grad_list_A.append(grad)

                elif "lora_B" in name:
                    grad = param.grad.detach()
                    grad_list_B.append(grad)

            norm_A, norm_B = None, None

            if grad_list_A:
                all_grads_A = torch.cat(grad_list_A)
                norm_A = all_grads_A.norm(norm_type).item()

            if grad_list_B:
                all_grads_B = torch.cat(grad_list_B)
                norm_B = all_grads_B.norm(norm_type).item()

            return norm_A, norm_B


        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            # opt_params = {id(p): i for i, g in enumerate(ctx.optimizer.param_groups) for p in g["params"]}
            # for n,p in ctx.model.named_parameters():
            #     if "lora_A" in n:
            #         print(n, "requires_grad", p.requires_grad, 
            #             "in_opt", id(p) in opt_params, 
            #             "opt_group", opt_params.get(id(p), None))
            #         if id(p) in opt_params:
            #             gi = opt_params[id(p)]
            #             print("  -> lr:", ctx.optimizer.param_groups[gi].get("lr"),
            #                 "weight_decay:", ctx.optimizer.param_groups[gi].get("weight_decay", 0))

            # for n,p in ctx.model.named_parameters():
            #     if "lora_A" in n:
            #         if p.grad is not None:
            #             cos_sim = torch.nn.functional.cosine_similarity(
            #                 p.grad.view(1, -1), p.data.view(1, -1)
            #             )
            #             print(n, "grad_norm:", p.grad.norm().item(),
            #                     "param_norm:", p.data.norm().item(),
            #                     "cos_sim(grad,param):", cos_sim.item())
            #             # print(n, "param_norm:", p.data.norm().item())
            #             break

            # for n,p in ctx.model.named_parameters():
            #     if "lora_B" in n:
            #         if p.grad is not None:
            #             cos_sim = torch.nn.functional.cosine_similarity(
            #                 p.grad.view(1, -1), p.data.view(1, -1)
            #             )
            #             # print(n, "grad_norm:", p.grad.norm().item(),
            #             #         "param_norm:", p.data.norm().item(),
            #             #         "cos_sim(grad,param):", cos_sim.item())
            #             print(n, "param_norm:", p.data.norm().item())
            #             break


            # added by me, for LoRA
            # norm_grad_A, norm_grad_B = get_lora_AB_grad_norm(ctx.model)
            # norm_A, norm_B = get_lora_matrices_norm(ctx.model)
            # print(f"LoRA A norm: {norm_grad_A}")
            # print(f"LoRA A grad norm: {norm_A}")

            p_before_dict = {}

            # Lưu giá trị param trước khi update
            for n, p in ctx.model.named_parameters():
                if "lora_A" in n and p.requires_grad:
                    p_before_dict[n] = p.detach().clone()

            ctx.optimizer.step()

            count = 0

            for n, p in ctx.model.named_parameters():
                if "lora_A" in n and p.requires_grad:
                    p_after = p.detach().clone()

                    # Compare changes after-before optimization step
                    # diff_norm = (p_after - p_before_dict[n]).norm().item()
                    # if diff_norm == 0:
                    #     count += 1
                    # count += 1
                    # print(f"{n} | Before norm: {p_before_dict[n].norm().item()} "
                    #     f"| After norm: {p_after.norm().item()} "
                    #     f"| Δ norm: {diff_norm}")

            # print(f"Number of LoRA A parameters with no change: {count}")

            # norm_A, norm_B = get_lora_matrices_norm(ctx.model)
            # print(f"LoRA A norm after step: {norm_A}")
            # print(f"LoRA B norm after step: {norm_B}")
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return
        
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate, use extend not append
        ctx.ys_true.extend(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.extend(ctx.y_pred.detach().cpu().numpy())
        
    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss
        }
        # added by me, for GLUE
        glue_metric = load_metric('glue', ctx.cfg.data.type.split('@')[0], trust_remote_code=True)
        # logger.info(glue_metric)
        eval_metric = glue_metric.compute(predictions=ctx.ys_pred, references=ctx.ys_true)
        # for k, v in eval_metric.items():
        #     eval_results[f'{ctx.cur_split}_{k}'] = v
        if 'accuracy' in eval_metric:
            eval_results[f'{ctx.cur_split}_accuracy'] = eval_metric['accuracy']
        
        setattr(ctx, 'eval_metrics', eval_results)
        
        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_glue_trainer(trainer_type):
    if trainer_type == 'gluetrainer':
        trainer_builder = GLUETrainer
        return trainer_builder


register_trainer('gluetrainer', call_glue_trainer)
