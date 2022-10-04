# acl4vc

PyTorch code for manuscript [*Adaptive Curriculum Learning for Video Captioning*](https://ieeexplore.ieee.org/document/9737531).

## Install

### Code

Create the virtual environment with conda and clone the code:

```bash
conda create -n cap python==3.7
conda activate cap
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm psutil h5py PyYaml wget pandas nltk
pip install pytorch-lightning==1.3.4 torchmetrics==0.6.0

git clone https://github.com/asuith/acl4vc
```

The main code for ideas behind our manuscript is `Sampler` Class and `get_idx_by_sorted_difficulty` function 
`misc/utils.py``' 
### Download

Here is the [OneDrive share](https://1drv.ms/u/s!ApDoTkmqHk3FiI1vaODLLO_v4nGwlQ?e=8euyfv) for the video-caption data feature and pretrained model weights, which are explained in the following.

#### Data

The code runs on two dataset, namely, MSVD and MSR-VTT. And the video features are extracted by pr-etrained ResNet-101 and ResNeXt-101, along with caption informations. 
The video difficulty (in the manuscript) is pre-calculated and shared in the file `vid_to_log_difficulty.pkl`.
It should be placed in somewhere to your liking (assume the folder is named `base_data_path`).

* Following the structure below to place corpora and feature files:
    ```
    └── base_data_path
        ├── MSRVTT
        │   ├── feats
        │   │   ├── image_resnet101_imagenet_fps_max60.hdf5
        │   │   └── motion_resnext101_kinetics_duration16_overlap8.hdf5
        │   ├── info_corpus.pkl
        │   ├── refs.pkl
        │   └── vid_to_difficulty.pkl
        └── Youtube2Text
            ├── feats
            │   ├── image_resnet101_imagenet_fps_max60.hdf
            │   └── motion_resnext101_kinetics_duration16_overlap8.hdf5
            ├── vid_to_difficulty.pkl
            ├── info_corpus.pkl
            └── refs.pkl
    ```
**Please remember to modify `base_data_path` in [config/Constants.py](config/Constants.py) to the actual directory**


#### Pretrained Models (optional)

We have provided the model weights trained on MSVD or MSRVTT. Download from the above link and extract to the `experiments` folder.

* Following the structure below to place pre-trained models (auxiliary files are ommited):
    ```
    └─experiments
        ├─MSRVTT
        │  ├─ARB_CL
        │  │  ├─acl
        │  │  │  └─best.ckpt
        │  │  │
        │  │  ├─baseline
        │  │  │  └─best.ckpt
        │  │  │
        │  │  └─pcl
        │  │      └─best.ckpt
        ......
    ```

To run the pretrained model, please see Test section below.


## Train

To train the model for yourself, you can pass in the arguments in the command line, or the pass in the `method_name` and change the responding file `config/methods.yaml` (or use both).

```
python train.py --default --dataset `dataset_name` --method `method_name`

# For example:

CUDA_VISIBLE_DEVICES=0 python train.py --save_csv --dataset MSVD \
    --method ParAhLSTMat_CL --sample_method metric \
    --difficulty_type video-rarity --slide \
    --slide_type middleExpand --start_mid 0.2 \
    --scope me02-v2-metric

CUDA_VISIBLE_DEVICES=0 python train.py --dataset MSRVTT \
    --method ARB --scope baseline
```

Checkout `opts.py` for the more arguments.

## Test

To test a trained model, using the following command with replaced `dataset_name` and `cp`.

```bash
CUDA_VISIBLE_DEVICES=0 python translate.py -gpus 1 --mode test -cp `actual model weight path` --save_csv

# For example:

CUDA_VISIBLE_DEVICES=0 python translate.py -gpus 1 --mode test \
    -cp experiments/MSRVTT/ARB_CL/acl/best.ckpt --save_csv

CUDA_VISIBLE_DEVICES=0 python translate.py -gpus 1 --mode test \
    -cp ../ACL4VC/model_weights/MSRVTT/ARB_CL/acl/best.ckpt --save_csv
```

## Acknowledgement
The code is modified from repo [yangbang18/Non-Autoregressive-Video-Captioning](https://github.com/yangbang18/Non-Autoregressive-Video-Captioning). Many thanks!

## Bibtex

Please consider citing our work if you find it helpful:

```
@article{DBLP:journals/access/LiYZ22,
  author    = {Shanhao Li and
               Bang Yang and
               Yuexian Zou},
  title     = {Adaptive Curriculum Learning for Video Captioning},
  journal   = {{IEEE} Access},
  volume    = {10},
  pages     = {31751--31759},
  year      = {2022},
  url       = {https://doi.org/10.1109/ACCESS.2022.3160451},
  doi       = {10.1109/ACCESS.2022.3160451},
}
```
