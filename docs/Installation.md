## Installation

#### 1. Clone this repository
```bash
git clone git@github.com:biansy000/ChatGarment.git
cd ChatGarment
```

#### 2. Install Dependencies
If you are not using Linux, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

```Shell
conda create -n chatgarment python=3.10 -y
conda activate chatgarment
pip install --upgrade pip  # enable PEP 660 support
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

#### 3. Install [GarmentCodeRC](https://github.com/biansy000/GarmentCodeRC)
Follow installation instructions in its repository.


#### 4. Download Pretrained Weights
Put the [Pretrained weights](https://sjtueducn-my.sharepoint.com/:u:/g/personal/biansiyuan_sjtu_edu_cn/EQayoB8ie7ZIsFrjLWdBASQBFexZHXcGjrS6ghgGCjIMzw?e=o60Y65) to ``checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin``.

#### 5. Update Paths in Code
Modify the following lines in relevant Python files:
```Python
sys.path.insert(1, '/is/cluster/fast/sbian/github/chatgarment_private') # path of the current ChatGarment repo
sys.path.insert(1, '/is/cluster/fast/sbian/github/GarmentCodeV2/') # path of GarmentCodeRC repo
```
Replace with their actual local paths.

#### 6. Add Soft Link 
Add the softlink of ``assets`` folder in ``GarmentCodeRC`` repo:
```Shell
ln -s path_to_garmentcode_assets assets
```
