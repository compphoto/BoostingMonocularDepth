## Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging 

This repository contains an implementation of our CVPR2021 publication:
>Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging. 
S. Mahdi H. Miangoleh, Sebastian Dille, Long Mai, Sylvain Paris, Yağız Aksoy.
[Main pdf](http://yaksoy.github.io/papers/CVPR21-HighResDepth.pdf),
[Supplementary pdf](http://yaksoy.github.io/papers/CVPR21-HighResDepth-Supp.pdf),
[Project Page](http://yaksoy.github.io/highresdepth/).

![Teaserimage](http://yaksoy.github.io/images/hrdepthTeaser.jpg)



### Change log:

* [Google Colaboratory notebook](./Boostmonoculardepth.ipynb) is now available.  [June 2021] 
* Merge net training dataset generation [instructions](./dataset_prepare/mergenet_dataset_prepare.md) is now available. [June 2021] 
* Bug fix. [June 2021]



## Setup

We Provided the implementation of our method using [MiDas-v2][1] and [SGRnet][2] as the base.

### Environments
Our mergenet model is trained using torch 0.4.1 and python 3.6 and is tested with torch<=1.8.

Download our mergenet model weights from [here](https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth) and put it in 
> .\pix2pix\checkpoints\mergemodel\latest_net_G.pth

To use [MiDas-v2][1] as base:
Install dependancies as following:
```sh
conda install pytorch torchvision opencv cudatoolkit=10.2 -c pytorch
conda install matplotlib
conda install scipy
conda install scikit-image
```
Download the model weights from [MiDas-v2][1] and put it in 
> ./midas/model.pt

```sh
activate the environment
python run.py --Final --data_dir PATH_TO_INPUT --output_dir PATH_TO_RESULT --depthNet 0
```

To use [SGRnet][2] as base:
Install dependancies as following:
```sh
conda install pytorch=0.4.1 cuda92 -c pytorch
conda install torchvision
conda install matplotlib
conda install scikit-image
pip install opencv-python
```
Follow the official [SGRnet][2] repository to compile the syncbn module in ./structuredrl/models/syncbn.
Download the model weights from [SGRnet][2] and put it in 
> ./structuredrl/model.pth.tar

```sh
activate the environment
python run.py --Final --data_dir PATH_TO_INPUT --output_dir PATH_TO_RESULT --depthNet 1
```

Different input arguments can be used to generate R0 and R20 results as discussed in the paper. 

```python
python run.py --R0 --data_dir PATH_TO_INPUT --output_dir PATH_TO_RESULT --depthNet #[0or1]
python run.py --R20 --data_dir PATH_TO_INPUT --output_dir PATH_TO_RESULT --depthNet #[0or1]
```

To generate the results with *CV.INFERNO* colormap use **--colorize_results** like the sample below:

```python
python run.py --colorize_results --Final --data_dir PATH_TO_INPUT --output_dir PATH_TO_RESULT --depthNet #[0or1]
```

### Evaluation
Fill in the needed variables in the following matlab file and run:
>./evaluation/evaluatedataset.m

* **estimation_path** : path to estimated disparity maps
* **gt_depth_path** : path to gt depth/disparity maps
* **dataset_disp_gttype** : (true) if ground truth data is disparity and (false) if gt depth data is depth.
* **evaluation_matfile_save_dir** : directory to save the evalution results as .mat file. 
* **superpixel_scale** : scale parameter to run the superpixels on scaled version of the ground truth images to accelarate the evaluation. use 1 for small gt images.


### Training

Navigate to [dataset preparation instructions](./dataset_prepare/mergenet_dataset_prepare.md) to download and prepare the training dataset. 

```sh
python ./pix2pix/train.py --dataroot DATASETDIR --name mergemodeltrain --model pix2pix4depth --no_flip --no_dropout
```
```sh
python ./pix2pix/test.py --dataroot DATASETDIR --name mergemodeleval --model pix2pix4depth --no_flip --no_dropout
```


## Citation

This implementation is provided for academic use only. Please cite our paper if you use this code or any of the models.
```
@INPROCEEDINGS{Miangoleh2021Boosting,
author={S. Mahdi H. Miangoleh and Sebastian Dille and Long Mai and Sylvain Paris and Ya\u{g}{\i}z Aksoy},
title={Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging},
journal={Proc. CVPR},
year={2021},
}
```

## Credits

The "Merge model" code skeleton (./pix2pix folder) was adapted from the [pytorch-CycleGAN-and-pix2pix][3] repository. 

For MiDaS and SGR inferences we used the scripts and models from [MiDas-v2][1] and [SGRnet][2] respectively (./midas and ./structuredrl folders). 

Thanks to [k-washi](https://github.com/k-washi) for providing us with a Google Colaboratory notebook implementation.

[1]: https://github.com/intel-isl/MiDaS/tree/v2
[2]: https://github.com/KexianHust/Structure-Guided-Ranking-Loss
[3]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

