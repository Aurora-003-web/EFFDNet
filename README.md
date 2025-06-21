# EFFDNet

This repository is for our paper "EFFDNet: A Scribble-Supervised Medical Image Segmentation Method with Enhanced Foreground Feature Discrimination"

## Requirements
Some important required packages include:

* Pytorch version >=0.4.1.

* TensorBoardX

* Python == 3.7

* Efficientnet-Pytorch

* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy，Batchgenerators ......

## Usage

### 1、Clone the repo;
```
git clone https://github.com/Aurora-003-web/EFFDNet.git
```

### 2、Data Preparation;

The dataset can be downloaded from:

[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)


[ISBI-MR-Prostate-2013 dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21267207)


### 3、Train the model;

```
cd EFFDNet/code

CUDA_VISIBLE_DEVICES=3 python -u train_weakly_supervised_2D.py --fold fold1 --num_classes 3 --root_path ../data/Prostate --exp Prostate/WeaklySeg --max_iterations 60000 --batch_size 12 --use_aa --use_contrast
```
### 4、Test the model;
```
cd EFFDNet/code

CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --sup_type scribble/label --exp ACDC/the trained model fold --model unet
```
Our code is based on the [WSL4MIS](https://github.com/HiLab-git/WSL4MIS). Thanks for these authors for their valuable works.
