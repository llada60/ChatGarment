#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# from transformers.generation.utils import *

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class GarmentGPTFloat50ForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, **kwargs,):
        super(GarmentGPTFloat50ForCausalLM, self).__init__(config)

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
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

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
        # input_ids_backup: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is not None:
            input_ids = None

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
            image_sizes=image_sizes,
            return_dict=return_dict
        )

        output_hidden_states = output.hidden_states

        if not inference:
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )

            padded_len = output_hidden_states[-1].shape[1] - seg_token_mask.shape[1]
            seg_token_mask = torch.cat(
                    [torch.zeros((seg_token_mask.shape[0], padded_len)).bool().cuda(), seg_token_mask],
                    dim=1,
                )
        
        else:
            #### inference
            return output


        if seg_token_mask.sum() > 0:
            last_hidden_state = self.float_layer(output_hidden_states[-1])

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
            text_embeddings = torch.cat(pred_embeddings, dim=0)

            if float_labels is not None:
                float_labels = torch.cat(float_labels, dim=0).type(text_embeddings.dtype)
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

        output_ids = output.logits.argmax(-1)

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


    
######################## ?
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, GarmentGPTFloat50ForCausalLM)
