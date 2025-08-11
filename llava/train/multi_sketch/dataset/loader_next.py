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
import bisect
import os
import glob
from llava.train.sketch.args.argument import DataArguments
from .preprocess_next import *
from .utils import *
from llava.mm_utils import process_images

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                            sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 max_len=-1,
                 random_imgaug=False,
                 has_sewing_pattern=False):
        super(LazySupervisedDataset, self).__init__()
        # print('LazySupervisedDataset', LazySupervisedDataset, data_path)
        list_data_dict = json.load(open(data_path, "r"))
        self.has_sewing_pattern = has_sewing_pattern

        if max_len > 0:
            list_data_dict = list_data_dict[:max_len]
        # print('list_data_dict', list_data_dict)

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.random_imgaug = random_imgaug

        try:
            if not 'sample_prob' in list_data_dict[0]:
                self.probs = np.ones(len(list_data_dict))

            else:
                self.probs = [
                    item['sample_prob'] for item in list_data_dict
                ]
                for i in range(len(self.probs)):
                    if self.probs[i] > 0.15 and self.probs[i] < 0.35:
                        self.probs[i] = 0.05
                    
                    elif self.probs[i] < 0.15:
                        self.probs[i] = 0.02

                self.probs = np.array(self.probs)
        except:
            print("data path ", data_path)
            print(type(list_data_dict))
            print(len(list_data_dict))

        self.probs = self.probs / self.probs.sum()
        self.cumsum = np.cumsum(self.probs)

        with open('docs/all_float_paths.json', 'r') as f:
            all_float_paths = json.load(f)

        self.all_float_paths_dict = {
            item: idx for idx, item in enumerate(all_float_paths)
        }

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'sketch_path' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'sketch_path' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = bisect.bisect_left(self.cumsum, random.random())
        sources = copy.deepcopy(self.list_data_dict[i])

        ################ prompt augmentation ################
        prompt = sources["conversations"][0]["value"] # human prompt
        prompt = change_prompt(prompt[:])
        # print("original prompt: ", prompt)
        sources["conversations"][0]["value"] = prompt

        if self.has_sewing_pattern:
            answer = sources["conversations"][1]["value"] # gpt answer
            answer, indices = change_answer(answer[:], self.all_float_paths_dict, True) 
            sources["conversations"][1]["value"] = answer

        ################ add trivial image if no image ################
        rand_flag = True
        if '<image>' not in prompt:
            sources["conversations"][0]["value"] = '<image>\n' + prompt
            self.list_data_dict[i]['sketch_path'] = "docs/images/black_img.jpg"
            sources["sketch_path"] = "docs/images/black_img.jpg"
            rand_flag = False
        else:
            sources["conversations"][0]["value"] = '<image>\n <image>\n <image>\n' + sources["conversations"][0]["value"]
        # print("final prompt: ", sources["conversations"][0]["value"])
        # print("sketch_path: ", self.list_data_dict[i]['sketch_path'])
        raise False
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        has_floats = False
        if 'all_floats' in sources[0]:
            has_floats = True
            all_floats, all_floats_weight = generate_all_float_labels(sources[0]['all_floats'], indices)
        
        if 'sketch_num' in sources[0]:
            sketch_num = sources[0]['sketch_num']
            # print("sketch_num: ", sketch_num)
            # sketch_num = len(sources[0]['sketch_path'])

            image_files = sources[0]['sketch_path']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            model = self.data_args.model

            # images = [convert_rgba_to_rgb_with_white_bg(os.path.join(image_folder, image_file)) for image_file in image_files]
            images = []
            image_tensors = []
            for image_file in image_files:
                image_file = image_file.strip()
                if not os.path.exists(image_file):
                    print(f"Image file {image_file} does not exist.")
                    continue
                image = convert_rgba_to_rgb_with_white_bg(os.path.join(image_folder, image_file))

                if self.random_imgaug and rand_flag:
                    image = autocrop(image)
                elif rand_flag:
                    image = autocrop(image, matrix_HW_pct_range=[0.8, 0.96])

                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image_tensor = process_images([image], processor, model.config)
                # print("image_tensor.shape", image_tensor.shape)
                image_tensors.append(image_tensor)
            # print("image_tensor length: ", len(image_tensors))
            images = [_image.to(dtype=torch.float16) for _image in image_tensors]
                
            # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # images.append(image)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('sketch_num' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        # print('sketch_path', image.shape) # image torch.Size([3, 336, 336])
        if 'sketch_num' in self.list_data_dict[i]:
            # data_dict['image'] = image
            assert sketch_num == len(image_files)
            print("images length: ", len(images))
            assert sketch_num == len(images)
            assert len(images)>=4
            selected_indices = random.sample(range(len(images)), 4)
            selected_images = [images[idx] for idx in selected_indices] 
            selected_paths = [os.path.join(image_folder, image_files[idx]) for idx in selected_indices]
            data_dict['images'] = torch.stack(selected_images, dim=0)[1]
            data_dict['image_paths'] = selected_paths
            print("data_dict['images'].shape: ", data_dict['images'].shape) # [3, 336, 336] for single sketch; for now torch.Size([4, 1, 3, 384, 384])
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(4, 3, crop_size['height'], crop_size['width'])
            data_dict['image_paths'] = ''
        
        if has_floats:
            data_dict['all_floats'] = torch.tensor(all_floats).float().reshape(-1)
            data_dict['float_weight'] = torch.tensor(all_floats_weight).float().reshape(-1)
        else:
            data_dict['all_floats'] = torch.zeros(0)
            data_dict['float_weight'] = torch.zeros(0)

        return data_dict

class LazySupervisedDatasetCmb(Dataset):
    def __init__(self, data_path_list: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 max_len=-1):
        super(LazySupervisedDatasetCmb, self).__init__()
        
        prob_dict = {
            "sewing_pattern_img": 0.4,
            "sewing_pattern_text": 0.2,
            "sewing_pattern_imgtext": 0.4
        }

        self.datasets = []
        self.ratios = []
        for k, v in data_path_list.items():
            prob = prob_dict[k] / len(v)
            has_sewing_pattern = True if 'sewing_pattern' in k else False
            random_imgaug = True if k != 'ift' else False
            self.datasets.extend([
                LazySupervisedDataset(
                    data_path=v_i,
                    tokenizer=tokenizer, data_args=data_args, max_len=max_len,
                    random_imgaug=random_imgaug,
                    has_sewing_pattern=has_sewing_pattern) for v_i in v
            ])

            self.ratios.extend([prob] * len(v))

        self.dataset_num = len(self.datasets)
        assert len(self.ratios) == self.dataset_num
        assert np.abs(np.sum(self.ratios) - 1) < 1e-3

        self.ratio_cumsum = np.cumsum(self.ratios)
        self.lengths = [len(self.datasets[i]) for i in range(self.dataset_num)]
        print('self.lengths', self.lengths)

    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        myrandom = random.Random(i * 1000 + random.randint(0, 1000))
        p = myrandom.random()
        for j in range(self.dataset_num):
            if p < self.ratio_cumsum[j]:
                idx = myrandom.randint(0, self.lengths[j] - 1)
                return self.datasets[j][idx]