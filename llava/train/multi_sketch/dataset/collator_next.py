from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX

import torch
import transformers

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        image_paths = [instance['image_paths'] for instance in instances]
        all_floats = [instance['all_floats'] for instance in instances]
        float_weight = [instance['float_weight'] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            image_paths=image_paths,
            float_labels=all_floats,
            float_weight=float_weight
        )
        # if input_ids is None:
        #     print(image_paths)
        #     raise False

        if 'images' in instances[0]:
            all_images = [instance['images'] for instance in instances]  # [N, 4, 3, H, W]
            
            if all(x is not None and x.shape == all_images[0].shape for x in all_images):
                batch['images'] = torch.stack(all_images)  # Stack N instances of [4, 3, H, W] into [N, 4, 3, H, W]
            else:
                batch['images'] = all_images  


        return batch