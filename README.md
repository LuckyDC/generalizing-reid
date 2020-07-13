# Generalizing Person Re-identification
Implementation of ECCV2020 paper "Generalizing Person Re-Identification by Camera-Aware Instance Learning and Cross-Domain Mixup"

## Dependencies
* python 3.5
* pytorch 1.5
* ignite
* apex

## Preparation
Download and extract Market-1501, DukeMTMC-reID, CUHK03 and MSMT17. 
Replace the root paths of corresponding datasets in the config file `configs/default/dataset.py`.


## Train
```shell script
bash train.sh GPU_ID_0,GPU_ID_1 PATH_TO_YOUR_YAML_FILE
``` 
Our code is validated under 2-GPUs setting.
GPU_ID_0 and GPU_ID_1 is the indices of the selected GPUs.
PATH_TO_YOUR_YAML_FILE is the path to your config yaml file.
We also offer the template of config file `configs/duke2market.yml`, `configs/market2duke.yml`, `configs/single_domain.yml`. 
You can optionally adjust the hyper-parameters in the config yaml file.

## Test
In our code, the model is evaluated on the target domain at intervals automatically.
You can also evaluate the trained model manually by running:
```shell script
python3 eval.py GPU_ID PATH_TO_CHECKPOINT_FILE [--dataset {market,duke,cuhk,msmt}]
```










