## Merge net training dataset preparation
### Download the datasets
Download rgb images of Ibims1-core-raw from [Ibims1 official webpage](https://www.bgu.tum.de/en/lmf/research/datasets/ibims-1/).

Download rgb images of Middleburry2014 "10 evaluation training sets with GT" + "13 additional datasets with GT
" from [Middleburry-2014 official webpage](https://vision.middlebury.edu/stereo/data/scenes2014/) . 

### Folder structure
Folder structure should be as below:

```bash
root_dir to datasets. 
|----middleburry
|    |----rgb
|         |----{rgb images.*}
|----ibims1
|    |----rgb
|         |----{rgb images.*}
```

### Setup

Set the "root_dir" parameter in bash
```bash
root_dir=''
```
Also, edit the root_dir parameter inside "./dataset_prepare/ibims1_prepare.m" and ""./dataset_prepare/generatecrops.m""exactly the same as the $root_dir parameter you've used above.


#### Step 1 : Remove not selected images from ibims1 dataset
```bash
cd ./dataset_prepare
## current dir : ./dataset_prepare
ibims1_prepare.m
```

#### Step 2 : Generate whole-image estimations

Download the midas weights from [MiDas-v2](https://github.com/intel-isl/MiDaS/tree/v2) and put it in 
> ./mergnet_dataset_prepare/midas/model.pt

Use the same python envirmenment as the one instructed in [Main method instruction](/README.md) under using [MiDas-v2](https://github.com/intel-isl/MiDaS/tree/v2) as base section. 

Run the following commands to generate estimations:
```python
cd ./midas/
## current dir : ./dataset_prepare/midas
python run.py --res 384 --input_dir $root_dir/ibims1/rgb --output_dir $root_dir/ibims1/whole_low_est
python run.py --res 672 --input_dir $root_dir/ibims1/rgb --output_dir $root_dir/ibims1/whole_high_est
python run.py --res 384 --input_dir $root_dir/middleburry/rgb --output_dir $root_dir/middleburry/whole_low_est
python run.py --res 672 --input_dir $root_dir/middleburry/rgb --output_dir $root_dir/middleburry/whole_high_est

```

#### Step 3 : Generate rgb, proxy ground truth and low resolution estimations of the patches
```bash
cd .. 
## current dir : ./dataset_prepare
create_crops.m
```
Allow ~20 minutes for the script execution to finish. 

#### Step 4 : Generate the patch estimations for high res input of the network
```python
cd ./midas/
## current dir : ./dataset_prepare/midas
python run.py --res 672 --input_dir $root_dir/mergenetdataset/train/rgb --output_dir $root_dir/mergenetdataset/train/inner
python run.py --res 672 --input_dir $root_dir/mergenetdataset/test/rgb --output_dir $root_dir/mergenetdataset/test/inner
```

Dataset is complete and is located at "$root_dir/mergenetdataset"
