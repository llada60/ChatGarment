# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import shutil
import time

import numpy as np
import torch
import random

import transformers
import tokenizers

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
import deepspeed
from functools import partial
from easydict import EasyDict as edict
from llava.garmentcode_utils import change_prompt, change_answer

from PIL import Image
from llava.lisa_utils import AverageMeter, ProgressMeter, dict_to_cuda, Summary
from torch.utils.tensorboard import SummaryWriter
import tqdm
import shutil
from llava.json_fixer import repair_json
import os
import glob
from dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset, LazySupervisedDatasetCmb
from llava.train.sketch.args.argument import DataArguments, ModelArguments, TrainingArguments

local_rank = None
os.environ["MASTER_PORT"] = "23480"

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def find_all_linear_names(model, lora_target_modules=['q_proj', 'v_proj']):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        'mm_projector', 'vision_tower', 'vision_resampler', 'float_layer'
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def validate_epoch(val_loader, model_engine, tokenizer, epoch, writer, args):
    return

def train_epoch(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    image_save_dir = os.path.join(args.log_dir, "training_images")
    os.makedirs(image_save_dir, exist_ok=True)
    
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    hmr_losses = AverageMeter("HMRLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            hmr_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    print('Start training ...')
    for global_step in range(args.steps_per_epoch):

        global_step_actual = epoch * args.steps_per_epoch + global_step
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            
            # continue

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            assert args.precision == "bf16"
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict.pop('image_paths')
            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            hmr_loss = output_dict["hmr_loss"]
            
            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            hmr_losses.update(hmr_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                hmr_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step_actual)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step_actual)
                writer.add_scalar(
                    "train/hmr_loss", hmr_losses.avg, global_step_actual
                )
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step_actual
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step_actual
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            hmr_losses.reset()
        
        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step_actual)
                
    return train_iter



def translate_args(model_args, data_args, training_args):
    args = edict(
        local_rank=local_rank,
        version=None,
        vis_save_path="./vis_output",
        precision="bf16",
        image_size=None,
        model_max_length=training_args.model_max_length,
        lora_r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        lora_target_modules=None,
        vision_tower=model_args.vision_tower,
        load_in_8bit=False,
        load_in_4bit=False,
        dataset=None,
        sample_rates=None,
        log_base_dir='./runs',
        exp_name="chatgarment_train_scratch",
        epochs=40,
        steps_per_epoch=500,
        batch_size=4,
        grad_accumulation_steps=8,
        val_batch_size=1,
        workers=8,
        lr=1e-4,
        ce_loss_weight=1.0,
        no_eval=False,
        eval_only=False,
        vision_pretrained=None,
        resume="",
        start_epoch=0,
        print_freq=1,
        gradient_checkpointing=training_args.gradient_checkpointing,
        beta1=0.9,
        beta2=0.999,

        use_mm_start_end=False
    )   

    return args

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, data_path_list) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDatasetCmb(tokenizer=tokenizer,
                                data_path_list=data_path_list,
                                data_args=data_args)
    
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path_eval,
                                data_args=data_args,
                                max_len=10)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return train_dataset, eval_dataset, data_collator



def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    args = translate_args(model_args, data_args, training_args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    bnb_model_from_pretrained_args = {}
    # writer = None
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    assert training_args.bits not in [4, 8]
    assert model_args.vision_tower is not None
    assert 'mpt' not in model_args.model_name_or_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")

    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    
    print('num_added_tokens', num_added_tokens, args.seg_token_idx)

    model = GarmentGPTFloat50ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        seg_token_idx=args.seg_token_idx,
        **bnb_model_from_pretrained_args
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False ####################### ?
    assert not model_args.freeze_backbone

    assert training_args.gradient_checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable() ###################### ？
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # tokenizer.add_tokens('<FLOAT>', special_tokens=True)
    assert model_args.version == "v1"
    if model_args.version in conversation_lib.conv_templates: # conv_vicuna_v1
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    lora_target_modules = find_all_linear_names(model)
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))
    model.print_trainable_parameters()

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "float_layer"]
            ]
        ):
            rank0_print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    ######################### ？

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    assert not model_args.tune_mm_mlp_adapter
    
    assert not training_args.freeze_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer) ############### ?

    model.print_trainable_parameters()

    data_path_list = {   
        "sewing_pattern_img": [ #['garment_id', 'sketch_num', 'conversations', 'all_floats', 'sample_prob', 'id', 'sketch_path']
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_restpose_img_v1.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_img_v2.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_img_v4.json',
        ],
        "sewing_pattern_text": [ # ['id', 'conversations', 'all_floats', 'float_mask', 'sample_prob']
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtext_v2.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtext_singlegarment_v2.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtext_v4.json',
        ],
        "sewing_pattern_imgtext": [ # ['garment_id', 'sketch_num', 'conversations', 'all_floats', 'float_mask', 'sample_prob', 'id', 'sketch_path']
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtextimg_v2.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtextimg_v3.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtextimg_v4.json',
            '/home/ids/liliu/data/ChatGarment/training/synthetic/data_detailtextimg_singlegarment_v2.json'
        ]
    }

    train_dataset, val_dataset, collate_fn = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args, data_path_list=data_path_list)
    
    rank0_print('train_dataset', len(train_dataset))
    # assert False

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, _, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=collate_fn,
        config=ds_config,
    )

    resume = os.path.join(args.log_dir, "ckpt_model")
    if args.resume:
        args.resume = resume
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    
    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train_epoch(
            train_loader,
            model_engine, 
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if True:
            best_score = 0.0
            cur_ciou = 0.0
            current_time = time.strftime("%m-%d_%H_%M", time.localtime())
            # save_dir follow by current time)
            save_dir = os.path.join(args.log_dir, "ckpt_model_epoch{}_{}".format(epoch, current_time))
            print("save_dir", save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


if __name__ == "__main__":
    train()
