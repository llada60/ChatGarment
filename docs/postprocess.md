# Postprocessing after ChatGarment Inference

ChatGarment may occasionally produce garments with incorrect lengths or widths from input images. To alleviate this, we provide a postprocessing method that refines garment sizes using a finite-difference-based approach. This process adjusts the garment length and width to better match the segmentation mask predicted by SAM (Segment Anything Model).

Assume that the input images are placed in the folder ``example_data/example_imgs``.

### Step 1. Garment Segmentation with Grounding-SAM
Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [segment-anything](https://github.com/facebookresearch/segment-anything) for segmentation. You can follow the installation instructions provided in [PuzzleAvatar](https://github.com/YuliangXiu/PuzzleAvatar/blob/main/scripts/install_dino_sam.sh)

Run the segmentation script:
```bash
python scripts/postprocess/grounding_sam.py --in_dir example_data/example_imgs --out_dir runs/example_eva_SAM
```

### Step 2. Human Pose and Shape Estimation with TokenHMR
Install [TokenHMR](https://github.com/saidwivedi/TokenHMR) for human pose estimation. Navigate to the TokenHMR directory:
```bash
cd PATH_TO_TOKENHMR
```

Next, modify ``demo.py`` by inserting the following code after [this line](https://github.com/saidwivedi/TokenHMR/blob/198645f7784a27a4df0eac32478b1e7bc3e13574/tokenhmr/demo.py#L116):
```python
                out_saved = out.copy()
                out_saved['pred_cam_t_full'] = pred_cam_t_full[n]
                out_saved['scaled_focal_length'] = scaled_focal_length
                for k, v in out_saved['pred_smpl_params'].items():
                    if isinstance(v, torch.Tensor):
                        out_saved['pred_smpl_params'][k] = v.detach().cpu().numpy()
                with open(os.path.join(args.out_folder, f'{img_fn}_{person_id}.pkl'), 'wb') as f:
                    pickle.dump(out_saved, f)
```

Then, run TokenHMR with the following command:
```bash
python tokenhmr/demo.py \
    --img_folder {PATH_TO_CCHATGARMENT}/runs/example_eva_SAM/imgs_upsampled \
    --batch_size=1 \
    --full_frame \
    --checkpoint data/checkpoints/tokenhmr_model_latest.ckpt \
    --model_config data/checkpoints/model_config.yaml \
    --out_folder {PATH_TO_CCHATGARMENT}/runs/example_eva_SAM/tokenhmr_output
```

### Step 3. Install Extra Packages
* Pytorch3D: Follow the official [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
* Chumpy: Install with pip: ``pip install chumpy``. 
    Then, comment out the following line in ``chumpy/__init__.py``: 
    
    ```python
    from numpy import bool, int, float, complex, object, unicode, str, nan, inf
    ```


### Step 4. Run the Postprocessing Script
Assume you ChatGarment inference results in ``runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final_eva/vis_new/``. Download the required [extra-data](https://drive.google.com/file/d/1QXezA3J6uXqWHGATmcw3jaYxRXY2Ctte/view?usp=sharing) and extract it to ``checkpoints/extra_data``. Now run the postprocessing script. For example, to process the image:``1aee14a8c7b4d56b4e8b6ddd575d1f561a72fdc75c43a4b6926f1655152193c6.png``, use:
```bash
python scripts/postprocess/postprocess.py --imgname 1aee14a8c7b4d56b4e8b6ddd575d1f561a72fdc75c43a4b6926f1655152193c6 \
    --img_dir runs/example_eva_SAM/imgs_upsampled \
    --inp_pose_params_dir runs/example_eva_SAM/tokenhmr_output \
    --garmentcode_dir runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/example_imgs_img_recon/vis_new/ \
    --saved_dir runs/example_eva_SAM/postprocess \
    --garment_seg_dir runs/example_eva_SAM/mask/
```
