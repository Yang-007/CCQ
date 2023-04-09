# CCQ
![image](code/network/CCQ.png)




This repo holds the pytorch implementation of CCQ:<br />

**CCQ: Cross-class Query Network for Partially Labeled Organ Segmentation.**

Our paper is accepted by AAAI 2023: The 37th AAAI Conference on Artificial Intelligence.

## Requirements
Python 3.7.6<br />
PyTorch==1.7.1<br />
[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)<br />

## Usage
### 0. Installation
* Clone this repo
```
git clone git@github.com:Yang-007/CCQ.git
cd CCQ
```
### 1. MOTS Dataset Preparation
Before starting, MOTS should be re-built from the several medical organ and tumor segmentation datasets

Partial-label task | Data source
--- | :---:
Liver | [data](https://competitions.codalab.org/competitions/17094)
Kidney | [data](https://kits19.grand-challenge.org/data/)
Hepatic Vessel | [data](http://medicaldecathlon.com/)
Pancreas | [data](http://medicaldecathlon.com/)
Colon | [data](http://medicaldecathlon.com/)
Lung | [data](http://medicaldecathlon.com/)
Spleen | [data](http://medicaldecathlon.com/)

* Download and put these datasets in `dataset/0123456/`. 
* Re-spacing the data by `python re_spacing.py`, the re-spaced data will be saved in `0123456_spacing_same/`.

The folder structure of dataset should be like

    dataset/0123456_spacing_same/
    ├── 0Liver
    |    └── imagesTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    |    └── labelsTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    ├── 1Kidney
    ├── ...
Prepared dataset is available in [data](https://drive.google.com/drive/folders/19vsGF2VlTsA4Z9VxpOIvbG7YRFR0YEgl?usp=sharing)

### 2. Model
Pretrained model is available in [checkpoint](https://drive.google.com/file/d/1YAvLnm_vujniOqo1VZVA5Rff68rqMBhO/view?usp=share_link)
### 3. Training
* cd `network/' and run 
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM train.py \
--img_attn_layers=4 \
--query_attn_layers=2 \
--num_query=32 \
--sgd=True \
--train_list='list/MOTS/MOTS_train.txt' \
--snapshot_dir='snapshots/CCQ_sgdlr1e2_2500_32q' \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=4 \
--num_epochs=2500 \
--num_cls=7 \
--output_channel=2 \
--start_epoch=0 \
--learning_rate=0.01 \
--num_workers=4 \
--random_scale=True \
--weight_std=True \
--random_mirror=True \
--itrs_each_epoch=60 \
>> train_result.txt &
```

### 4. Evaluation
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python evaluate.py \
--val_list='list/MOTS/MOTS_test.txt' \
--reload_from_checkpoint=True \
--reload_path='snapshots/CCQ_sgdlr1e2_2500_32q/MOTS_CCQ_snapshots.pth' \
--save_path='outputs32q/CCQ_sgdlr1e2_2500_32q' \
--input_size='64,192,192' \
--num_workers=2 \
--num_cls=7 \
--output_channel=2 \
--num_query=32 \
--weight_std=True \
>> evaluate.txt &
```

### 5. Post-processing

```python
nohup python postp_save.py --img_folder_path='outputs32q/CCQ_sgdlr1e2_2500_32q/' \
--postp_outputs='postp_outputs/CCQ_sgdlr1e2_2500_32q/' \
>>postp.txt &
```

## Acknowledgement

Part of code obtained from [DoDNet](https://git.io/DoDNet) codebase.











