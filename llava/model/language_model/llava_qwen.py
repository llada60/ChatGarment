from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_next_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, **kwargs,):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.model = LlavaQwenModel(config)
        self.last_dim = 76
        self.float_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.last_dim)
        )
        self.float_layer.train()
        for p in self.float_layer.parameters():
            p.requires_grad = True

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,# [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        float_labels: Optional[torch.FloatTensor] = None,
        float_weight: Optional[torch.FloatTensor] = None,
        inference: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(dtype=torch.float)
        print("attention_mask:", attention_mask)
        print("attention_mask dtype:", attention_mask.dtype)
        print("attention_mask unique values:", torch.unique(attention_mask))
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("max input_ids:", input_ids.max())
        print("vocab_size:", self.model.config.vocab_size)
        # """
        # attention_mask: WARNING: tokenization mismatch: 511 vs. 512. (ignored)
        # tensor([[ True,  True,  True,  ..., False, False, False],
        #         [ True,  True,  True,  ...,  True,  True,  True],
        #         [ True,  True,  True,  ..., False, False, False],
        #         [ True,  True,  True,  ..., False, False, False]], device='cuda:0')
        # attention_mask dtype: torch.bool
        # attention_mask unique values: tensor([False,  True], device='cuda:0')
        # input_ids shape: torch.Size([4, 660])
        # attention_mask shape: torch.Size([4, 660])
        # max input_ids: tensor(151647, device='cuda:0')
        # """
        # if inputs_embeds is None:
        #     (_, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        # else:
        if inputs_embeds is not None:
            input_ids = None
        
        output = super().forward(
            input_ids=input_ids, #FIXME: here the LLava-OV's input_ids is not the same structure as LLava, but should figure out and change the input style
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output_hidden_states = output.hidden_states

        if inference:
            return output
        # print(input_ids.shape) # [4,625]
        # print(inputs_embeds.shape) # [1, 3590, 3584]
        # === [SEG] Token Mask ===
        # FIXME: after previous bug, maybe here should change follow the new structure
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1), dtype=torch.bool, device=seg_token_mask.device)],
            dim=1
        )

        padded_len = output_hidden_states[-1].shape[1] - seg_token_mask.shape[1]
        if padded_len > 0:
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], padded_len), dtype=torch.bool, device=seg_token_mask.device), seg_token_mask],
                dim=1
            )

        if seg_token_mask.sum() > 0:
            last_hidden_state = self.float_layer(output_hidden_states[-1])  # shape: [bs, seq_len, last_dim]
            pred_embeddings = last_hidden_state[seg_token_mask]  # shape: [num_seg_tokens, last_dim]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs]
            seg_token_offset = seg_token_counts.cumsum(0)
            seg_token_offset = torch.cat([torch.tensor([0], device=seg_token_mask.device), seg_token_offset], dim=0)  # [bs+1]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            text_embeddings = torch.cat(pred_embeddings, dim=0)  # [total_seg_tokens, last_dim]

            if float_labels is not None:
                float_labels = torch.cat(float_labels, dim=0).to(dtype=text_embeddings.dtype)
                hmr_loss = torch.abs(text_embeddings.reshape(-1, self.last_dim) - float_labels.reshape(-1, self.last_dim))
                if float_weight is not None:
                    float_weight = torch.cat(float_weight, dim=0).to(dtype=text_embeddings.dtype)
                    hmr_loss = (hmr_loss * float_weight.reshape(-1, self.last_dim)).mean()
                else:
                    hmr_loss = hmr_loss.mean()
                    raise NotImplementedError
            else:
                hmr_loss = output.loss
        else:
            hmr_loss = output.loss * 0.0

        output_ids = output.logits.argmax(-1)

        if output.loss is None:
            return output
        return {
            "loss": output.loss + hmr_loss,
            "ce_loss": output.loss,
            "hmr_loss": hmr_loss,
            "predictions": output,
            "output_ids": output_ids
        }


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)