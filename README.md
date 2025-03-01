<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">ChatGarment: Garment Estimation, Generation and Editing via Large Language Models
 </h1>
  <div align="center">
    <img src="docs/images/teaser.png" alt="teaser" width="100%">
  </div>
</p> 


This is the implementation of ChatGarment. More details please check our 
[[Project Page](https://chatgarment.github.io/)].

ChatGarmen utilizes large vision-language models (VLMs) to automate the estimation, generation, and editing of 3D garments from images or text descriptions. 


## Applications


| ![](docs/images/img_recon.gif)  | <img src="docs/images/text_generation.png" width="2000"> |
| :--------------------: | :----------: |
| Image-based Reconstruction | Text-based Generation |
| ![](docs/images/video_edit.gif)  |  ![](docs/images/video_edit_2.gif)   |
| Text-based Editing | Text-based Editing |

## Install

#### Environment

1. Clone this repository and navigate to ChatGarment folder
```bash
git clone git@github.com:biansy000/ChatGarment.git
cd ChatGarment
```

2. Install Package

If you are not using Linux, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

4. Install [GarmentCodeRC](https://github.com/biansy000/GarmentCodeRC).

#### Pretrained weights and preparations
5. Put the [Pretrained weights](https://sjtueducn-my.sharepoint.com/:u:/g/personal/biansiyuan_sjtu_edu_cn/EQayoB8ie7ZIsFrjLWdBASQBFexZHXcGjrS6ghgGCjIMzw?e=o60Y65) to ``checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin``.

6. Change the following codes in *.py files:
```Python
sys.path.insert(1, '/is/cluster/fast/sbian/github/chatgarment_private') # path of the current repo
sys.path.insert(1, '/is/cluster/fast/sbian/github/GarmentCodeV2/') # path of GarmentCode repo
```
to their actual paths.

7. Add the softlink of ``assets`` folder in ``GarmentCodeRC`` repo:
```Shell
ln -s path_to_garmentcode_assets assets
```


## Model Training
```Shell
./scripts/v1_5/finetune_task_lora_garmentcode_wholebody_combineT2.sh
```

## Model Inference

#### 1. Image-based Reconstruction (CoT)
```Shell
# Run image based reconstruction with CoT for images in example_imgs/
# Detailed steps of the script:
# 1. Accepts an input image.
# 2. Utilizes ChatGarment Model to generate text prompts based on the image.
# 3. Sends the ChatGarment-generated text & input image to ChatGarment Model again.
# 4. Outputs the final GarmentCode sewing patterns.
./scripts/v1_5/evaluate_garment_v2_imgfolder_2step.sh example_imgs/
```


#### 2. Text-based Generation
```Shell
# Run text based generation for images in example_imgs/
# Detailed steps of the script:
# 1. Accepts an input image.
# 2. Utilizes GPT-4o to generate text prompts based on the image.
# 3. Sends the GPT-generated text exclusively to ChatGarment Model.
# 4. Outputs the final GarmentCode sewing patterns.
./scripts/v1_5/evaluate_garment_v2_imgfolder_1step_wopenai_simplified.sh example_imgs/
```

#### 3. Garment Editing
```Shell
# Evaluate on ``runs/hood_simulation_garmentcode_eva_pair/`` folder (will be released).
# Detailed steps of the script:
# 1. Accepts the sewing pattern & the editing commands.
# 2. Outputs the target sewing pattern.
./scripts/v1_5/evaluate_garment_v2_eva_edit.sh
```


#### 4. Generate 3D Garments Based on ChatGarment Output
```Shell
# Run garment stitching to get draped 3D garments
python run_garmentcode_sim.py --all_paths_json $(OUTPUT_DIR)
```

## Citation
```bibtex
@article{bian2024chatgarment,
  title={ChatGarment: Garment Estimation, Generation and Editing via Large Language Models},
  author={Bian, Siyuan and Xu, Chenghao and Xiu, Yuliang and Grigorev, Artur and Liu, Zhen and Lu, Cewu and Black, Michael J and Feng, Yao},
  journal={arXiv preprint arXiv:2412.17811},
  year={2024}
} 
```

## Acknowledgments 
This repository is built extensively on top of [LLaVA](https://github.com/haotian-liu/LLaVA) and [LISA](https://github.com/dvlab-research/LISA). 
