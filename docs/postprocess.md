# Postprocessing after ChatGarment Inference

ChatGarment may occasionally produce garments with incorrect lengths or widths from input images. To alleviate this, we provide a postprocessing method that refines garment sizes using a finite-difference-based approach. This process adjusts the garment length and width to better match the segmentation mask predicted by SAM (Segment Anything Model).

Assume that the input images are placed in the folder ``example_data/example_imgs``.

### Step 1. Garment Segmentation with Grounding-SAM
Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [segment-anything](https://github.com/facebookresearch/segment-anything) for segmentation. Run the segmentation script:
```bash
python scripts/postprocess/grounding_sam.py --in_dir example_data/example_imgs --out_dir runs/example_eva_SAM
```

### Step 2. Human Pose and Shape Estimation with TokenHMR
Install [TokenHMR](https://github.com/saidwivedi/TokenHMR) for human pose estimation. Navigate to the TokenHMR directory:
```bash
cd PATH_TO_TOKENHMR
```

Run TokenHMR with the following command:
```bash
python tokenhmr/demo.py \
    --img_folder {PATH_TO_CCHATGARMENT}/runs/example_eva_SAM/imgs_upsampled \
    --batch_size=1 \
    --full_frame \
    --checkpoint data/checkpoints/tokenhmr_model_latest.ckpt \
    --model_config data/checkpoints/model_config.yaml \
    --out_folder {PATH_TO_CCHATGARMENT}/runs/example_eva_SAM/tokenhmr_output
```


### Step 3. Run the Postprocessing Script
Assume you ChatGarment inference results in ``runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final_eva/vis_new/``. Downloaded the required [extra-data](https://drive.google.com/file/d/1QXezA3J6uXqWHGATmcw3jaYxRXY2Ctte/view?usp=sharing). Now run the postprocessing script for an image named ``exampleimg.png``:
```bash
python scripts/postprocess/postprocess.py --imgname exampleimg \
    --img_dir runs/example_eva_SAM/imgs_upsampled \
    --inp_pose_params_dir runs/example_eva_SAM/tokenhmr_output \
    --garmentcode_dir runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final_eva/vis_new/ \
    --saved_dir runs/example_eva_SAM/postprocess \
    --garment_seg_dir runs/example_eva_SAM/mask/
```
