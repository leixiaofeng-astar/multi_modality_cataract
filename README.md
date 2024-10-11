# multi_modality_cataract
To develop and evaluate the performances of Deep Learning algorithms in detecting visually significant cataract using retinal, slit view, and diffuse anterior segment photos.

# Prerequisite

## Hardware Resource Recommendations
- CPU: Intel Core or Xeon Serial 64 bits Processors (released in recent years)
- Memory: More than 16G
- Disk: More than 20G free space
- GPU: Not necessary

## User
A sudo user is required for running commands in following sections.

## Operating System
Recommend to use Ubuntu 20.04 LTS (64 bits), Ubuntu 22.04 LTS or later versioin.
The code also can run on the virtual environment in other Linux, windows and Mac OS operation system.

#### System should be updated to latest version:
```
sudo apt-get update
sudo apt-get upgrade -y

```

## Software
#### Reqired System Software Packages
```
apt-get install --no-install-recommends -y python3-pip
apt-get install ffmpeg libsm6 libxext6  -y 
apt-get install --assume-yes apt-utils
apt-get install -y build-essential swig
apt-get clean
```
pip recommend to be upgraded to latest version:
```
pip install --upgrade pip
```
If this is the 1st time to upgrade pip as normal user, logout and login will be required in order to use the new version **pip** installed in user home directory.

#### Required Python Packages
All required packages with specific versions are listed in file **requirements.txt**, run command to install:
```
pip install -r requirements.txt
```
You can also use virtual environment to setup the working environment, run command to install:
```
sudo apt install python3-venv
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
you will find 3 types of sample images under the folder:
	fundus_images
	diffuse_images
	slitlamp_images
You can run our models on your image. The supported image file format includes: png, bbmp, jpg, or tiff.

# Prediction

## Usage
```
usage: python3 main.py --input DATASET_DIR [--output OUTPUT_DIR] [--threshold THRESHOLD_VALUE] [-h]

options:
  --input DATASET_DIR         The input directory for dataset image files, must be specified.
  --output OUTPUT_DIR         The result output csv file directory, optional, default to *./outputs*.
  -h                          Show command line options.

examples:
  git clone https://github.com/SunnyAVT/visually_significant_cataract.git
  cd multi_modality_cataract
  python3 main.py --image_type fundus --input ./fundus_images/MS26132R_R.png  
  python3 main.py --image_type diffuse --input ./diffuse_images/OS000C33_MS25394.jpg  
  python3 main.py --image_type slitlamp --input ./slitlamp_images/OD000010_MS16616.jpg   

```

## Result
The prediction result will be shown as below:
```
python3 main.py --image_type fundus --input ./fundus_images/MS26132R_R.png 
/data/xiaofeng/retinal_cateract/cataract_paper/paper-env/lib/python3.8/site-packages/xgboost/compat.py:31: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
loading model: any048_class_model4ResNet18_xgb131-AUC9703.pth
loading any048_class_model4ResNet18_xgb131-AUC9703.pth model, time 0.28

Test loading model statedict: ResNet18
filename:./fundus_images/MS26132R_R.png, probability_0:0.9923532432876527, probability_1:0.007646756712347269
Cataract Validation is Over, please get your results in outputs/TestResult.csv !!!


$ python3 main.py --image_type diffuse --input ./diffuse_images/OS000C33_MS25394.jpg  
/data/xiaofeng/retinal_cateract/cataract_paper/paper-env/lib/python3.8/site-packages/xgboost/compat.py:31: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
Loaded pretrained weights for efficientnet-b3

Test loading model statedict: EfficientNet_b3
Loaded pretrained weights for efficientnet-b3

Test loading model statedict: any048_fold1_class_model4EfficientNet_b3_lr0.0001-AUC9438-statedict.pth
any048_fold1_class_model4EfficientNet_b3_lr0.0001-AUC9438-statedict.pth statedict model successfully loaded to CPU
filename:./diffuse_images/OS000C33_MS25394.jpg, probability_0:0.0008713603019714355, probability_1:0.9991286396980286
Cataract Validation is Over, please get your results in outputs/TestResult.csv !!!


$ python3 main.py --image_type slitlamp --input ./slitlamp_images/OD000010_MS16616.jpg   
/data/xiaofeng/retinal_cateract/cataract_paper/paper-env/lib/python3.8/site-packages/xgboost/compat.py:31: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index

Test loading model statedict: ResNet34

Test loading model statedict: any048_class_model4ResNet34_lr0.0001_2nd-AUC9338-statedict.pth
any048_class_model4ResNet34_lr0.0001_2nd-AUC9338-statedict.pth statedict model successfully loaded to CPU
filename:./slitlamp_images/OD000010_MS16616.jpg, probability_0:0.9999999998802167, probability_1:1.197832943944377e-10
Cataract Validation is Over, please get your results in outputs/TestResult.csv !!!
```
