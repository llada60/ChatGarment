'''
Copied from https://github.com/YuliangXiu/PuzzleAvatar/blob/main/multi_concepts/grounding_dino_sam.py
'''
import argparse
import os
import sys
import random
from PIL import Image
from glob import glob

import torch

GroundingDINO_dir = "/is/cluster/fast/sbian/github/PuzzleAvatar/thirdparties/GroundingDINO"
sys.path.insert(0, GroundingDINO_dir)

import base64
import json
import io
from typing import List

import cv2
import numpy as np
from groundingdino.util.inference import Model
from segment_anything import SamPredictor, sam_model_registry
from tqdm.auto import tqdm
from scipy import ndimage

from openai import OpenAI
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def enhance_class_name(class_names: List[str]) -> List[str]:
    return class_names



# Function to encode the image
def encode_image(image_path):
    buffer = io.BytesIO()
    # img = Image.open(image_path).convert('RGB')
    img = convert_rgba_to_rgb_with_white_bg(image_path)
    
    width, height = img.size
    if width > height:
        new_width = 800
        new_height = int((800 / width) * height)
    else:
        new_height = 800
        new_width = int((800 / height) * width)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:    # shrinking image
        interp = cv2.INTER_AREA
    else:    # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h    # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:    # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:    # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:    # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padColor
    )

    return scaled_img


def convert_rgba_to_rgb_with_white_bg_cv2(image_path):
    """Convert an RGBA image to an RGB image with a white background using OpenCV."""

    # Read the image with unchanged flag to keep alpha channel if present
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # Check if image has alpha channel
    if image.shape[2] == 4:
        # Split into color and alpha channels
        b, g, r, a = cv2.split(image)
        
        # Normalize alpha to range 0–1
        alpha = a.astype(float) / 255.0
        
        # Create a white background
        white_bg = np.ones_like(b, dtype=float) * 255

        # Blend each channel with white background
        r = r.astype(float) * alpha + white_bg * (1 - alpha)
        g = g.astype(float) * alpha + white_bg * (1 - alpha)
        b = b.astype(float) * alpha + white_bg * (1 - alpha)

        # Merge and convert back to uint8
        rgb_image = cv2.merge([b, g, r]).astype(np.uint8)
    else:
        rgb_image = image

    return rgb_image


def convert_rgba_to_rgb_with_white_bg(image_path):
    """Convert an RGBA image to an RGB image with a white background."""
    if not os.path.exists(image_path):
        image_path = image_path[:-4] + '.gif'

    # Open the image
    image = Image.open(image_path)

    # Check if the image is in RGBA mode
    if image.mode == 'RGBA':
        # Create a white background image of the same size as the original
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        
        # Paste the RGBA image onto the white background, using the alpha channel as the mask
        white_background.paste(image, mask=image.split()[3])  # Split to get the alpha channel
        image = white_background
    else:
        # If not RGBA, save it directly as an RGB image
        image = image.convert("RGB")
    
    return image


def gpt4v_captioning(img_dir, out_dir):

    if True:
        used_lst = sorted(os.listdir(img_dir))
        prompt = open("docs/prompts/gpt4v_prompt_garment_sam.txt", "r").read()

    images = [encode_image(os.path.join(img_dir, img_name)) for img_name in used_lst if img_name.endswith('.jpg') or img_name.endswith('.png')]
    if not os.path.exists(os.path.join(out_dir, 'gpt_responses')):
        os.makedirs(os.path.join(out_dir, 'gpt_responses'), exist_ok=True)

    saved_paths = [os.path.join(out_dir, 'gpt_responses', f'gpt_response_{i}_{img_name[:-4]}.json') for i, img_name in enumerate(used_lst)]
    
    client = openAI()
    
    results = {}
    for i, image in tqdm(enumerate(images)):
        response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {   
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}",
                                    "detail": "low"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

        result = response.choices[0].message.content
        with open(saved_paths[i], "w") as f:
            f.write(result)

        # results.append(result)
        results[ f'{i}_{used_lst[i][:-4]}' ] = result

    return results




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help="input image folder")
    parser.add_argument('--out_dir', type=str, required=True, help="output mask folder")
    parser.add_argument('--overwrite', action="store_true")
    opt = parser.parse_args()

    gpt_filename = "gpt4v_prompt.json"

    if not os.path.exists(f"{opt.out_dir}/mask"):
        os.makedirs(f"{opt.out_dir}/mask", exist_ok=True)

    if opt.overwrite:
        for f in os.listdir(f"{opt.out_dir}/mask"):
            os.remove(os.path.join(f"{opt.out_dir}/mask", f))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # paths
    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GroundingDINO_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        GroundingDINO_dir, "weights/groundingdino_swint_ogc.pth"
    )
    SAM_CHECKPOINT_PATH = os.path.join(GroundingDINO_dir, "weights/sam_vit_h_4b8939.pth")
    SAM_ENCODER_VERSION = "vit_h"

    # load models
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
    )
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    BOX_TRESHOLD = 0.20
    TEXT_TRESHOLD = 0.25

    if True:
        gpt4v_responses = gpt4v_captioning(opt.in_dir, out_dir=opt.out_dir)
        print(gpt4v_responses)

    image_dir = sorted(os.listdir(opt.in_dir))
    for imgid, img_name in enumerate(tqdm(image_dir)):
        CLASSES = [item.strip() for item in json.loads(gpt4v_responses[ f'{imgid}_{img_name[:-4]}' ]).keys()]
        CLASSES = ["person"] + CLASSES

        print(CLASSES)

        img_path = os.path.join(opt.in_dir, img_name)
        if not os.path.exists(os.path.join(opt.out_dir, 'imgs_upsampled')):
            os.makedirs(os.path.join(opt.out_dir, 'imgs_upsampled'), exist_ok=True)

        # image = cv2.imread(img_path)
        image = convert_rgba_to_rgb_with_white_bg_cv2(img_path)
        if True:
            image = resizeAndPad(image, (4096, 4096))
            cv2.imwrite(os.path.join(opt.out_dir, 'imgs_upsampled', img_name), image)

            # detect objects
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=enhance_class_name(class_names=CLASSES),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

            # convert detections to masks
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            
            print(detections.class_id)

            mask_dict = {}

            # if there is person in the image
            assert 0 in detections.class_id
            if True:
                person_masks = detections.mask[detections.class_id == 0]
                person_mask = (np.stack(person_masks).sum(axis=0) > 0).astype(np.uint8)
                
                cv2.imwrite(f"{opt.out_dir}/mask/{img_name[:-4]}_person.png", person_mask.astype(np.uint8) * 255)

                for mask, cls_id in zip(detections.mask, detections.class_id):
                    if cls_id is not None and cls_id != 0:
                        if np.logical_and(mask, person_mask).sum() / person_mask.sum() < 0.9:
                            mask_dict[cls_id] = mask_dict.get(cls_id, []) + [mask]

                mask_final = {}

                # stack all the masks of the same class together within the same image
                for cls_id, masks in mask_dict.items():
                    mask = np.stack(masks).sum(axis=0) > 0
                    mask_final[cls_id] = mask

                # remove the overlapping area
                for cls_id, mask in mask_final.items():
                    mask_other = np.zeros_like(mask)
                    other_cls_ids = list(mask_final.keys())
                    other_cls_ids.remove(cls_id)
                    for other_cls_id in other_cls_ids:
                        mask_other += mask_final[other_cls_id]
                    mask_final[cls_id] = mask * (mask_other == 0)

                    mask_area = mask_final[cls_id].shape[0] * mask_final[cls_id].shape[1]

                    if (mask_final[cls_id]).sum() / mask_area > 0.001:
                        if CLASSES[cls_id] not in ["eyeglasses", "glasses"]:
                            cv2.imwrite(
                                f"{opt.out_dir}/mask/{img_name[:-4]}_{CLASSES[cls_id]}.png",
                                mask_final[cls_id].astype(np.uint8) * 255
                                )
                    
                    else:
                        cv2.imwrite(
                                f"{opt.out_dir}/mask/{img_name[:-4]}_{CLASSES[cls_id]}_wrong.png",
                                mask_final[cls_id].astype(np.uint8) * 255
                                )

