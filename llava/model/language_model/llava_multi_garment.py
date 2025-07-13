from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class Multi_GarmentGPTFloat50ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, **kwargs, ):
        super(Multi_GarmentGPTFloat50ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None


        self.seg_token_idx = kwargs.pop("seg_token_idx")

        # self.float_layer = nn.Linear(config.hidden_size, 1)
        self.last_dim = 76
        self.float_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.last_dim)
        )
        self.float_layer.train()
        for p in self.float_layer.parameters():
            p.requires_grad = True


        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
        float_labels: Optional[torch.FloatTensor] = None,
        float_weight: Optional[torch.FloatTensor] = None,
        inference: Optional[bool] = False,        
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        inference: Option[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # if dpo_forward:
        #     outputs = self.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )

        #     hidden_states = outputs[0]
        #     logits = self.lm_head(hidden_states)
        #     return logits, labels

        # else:
        output = super().forward(
            input_ids=input_ids, # [batch_size, seq_len]
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels, # following Chatgarment
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images, # following Chatgarment
            image_sizes=image_sizes, # following Chatgarment
            return_dict=return_dict
        )

        output_hidden_states = output.hidden_states

        if not inference:
            # find the [SEG]  token, skip the first token
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx 
            # add a zero padding at the end to align with input_ids
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            # assure the seg_token_mask is the same length as output_hidden_states by adding 0 padding in the front
            padded_len = output_hidden_states[-1].shape[1] - seg_token_mask.shape[1]
            # True on [SEG], False on other idx.
            seg_token_mask = torch.cat(
                    [torch.zeros((seg_token_mask.shape[0], padded_len)).bool().cuda(), seg_token_mask],
                    dim=1,
                )
        
        else:
            #### inference
            return output

        if seg_token_mask.sum() > 0: # at least one [SEG] token
            # output_hidden_states.shape # [bs, seq_len, hidden_size]
            # seg_token_mask.shape # [bs, seq_len]
            last_hidden_state = self.float_layer(output_hidden_states[-1]) # [bs, seq_len, last_dim=76]

            pred_embeddings = last_hidden_state[seg_token_mask] # [num_seg_tokens, last_dim=76]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1) 
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            ) # [bs + 1, ]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])

            pred_embeddings = pred_embeddings_
            text_embeddings = torch.cat(pred_embeddings, dim=0) # all the [SEG] predictions, [num_seg_tokens, last_dim=76]

            if float_labels is not None:
                float_labels = torch.cat(float_labels, dim=0).type(text_embeddings.dtype)
                # last_dim corresponds to the clothing parameters
                hmr_loss = torch.abs(text_embeddings.reshape(-1, self.last_dim) - float_labels.reshape(-1, self.last_dim))
                if float_weight is not None:
                    float_weight = torch.cat(float_weight, dim=0).type(text_embeddings.dtype)
                    assert hmr_loss.shape == float_weight.reshape(-1, self.last_dim).shape, (hmr_loss.shape, float_weight.shape)
                    hmr_loss = (hmr_loss * float_weight.reshape(-1, self.last_dim)).mean()
                else:
                    hmr_loss = hmr_loss.mean()
                    raise NotImplementedError   
            else:
                hmr_loss = output.loss

        else:
            hmr_loss = output.loss * 0.0

        output_ids = output.logits.argmax(-1) # [bs, seq_len]

        if output.loss is None:
            return output

        return {
            "loss": output.loss + hmr_loss,
            "ce_loss": output.loss,
            "hmr_loss": hmr_loss,
            "predictions": output,
            'output_ids': output_ids
        }


    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        max_new_tokens=32,
        tokenizer=None,
    ):  
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                inference=True
            )
            # print('output_hidden_states', len(outputs.sequences[0]), outputs.sequences[0].shape, 
            #       len(outputs.hidden_states), outputs.hidden_states[-1][0].shape)
            # output_hidden_states 1001 torch.Size([1001]) 1000 torch.Size([1, 1, 4096])
            output_hidden_states = [item[-1] for item in outputs.hidden_states[1:]]
            output_hidden_states = torch.cat(output_hidden_states, dim=1)
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 2:] == self.seg_token_idx

            if seg_token_mask.sum() > 0:
                last_hidden_state = self.float_layer(output_hidden_states).reshape(1, -1, self.last_dim)
                # last_hidden_state = self.float_layer(output_hidden_states).reshape(1, -1, 1)
                pred_embeddings = last_hidden_state[seg_token_mask]

                seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
                seg_token_offset = seg_token_counts.cumsum(-1)
                seg_token_offset = torch.cat(
                    [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
                )

                pred_embeddings_ = []
                for i in range(len(seg_token_offset) - 1):
                    start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                    pred_embeddings_.append(pred_embeddings[start_i:end_i])
                pred_embeddings = pred_embeddings_
                
                ### -------------------------- SMPL decoder part: embedding -> SMPL parameters and corresponding losses -------------------------- ###
                text_embeddings = torch.cat(pred_embeddings, dim=0)
            else:
                text_embeddings = None

        return output_ids, text_embeddings, seg_token_mask


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs['inference'] = kwargs['inference']
        # inputs['input_ids_backup'] = input_ids

        return inputs

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


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)