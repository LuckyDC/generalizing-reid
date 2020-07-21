# Generalizing Person Re-identification
Implementation of ECCV2020 paper *Generalizing Person Re-Identification by Camera-Aware Instance Learning and Cross-Domain Mixup*.

## Dependencies
* python 3.6
* pytorch 1.5
* [apex](https://github.com/NVIDIA/apex)
* [ignite](https://github.com/pytorch/ignite)

## Preparation
Download and extract Market-1501, DukeMTMC-reID, CUHK03 and MSMT17. 
Replace the root paths of corresponding datasets in the config file `configs/default/dataset.py`.


## Train
```shell script
bash train.sh GPU_ID_0,GPU_ID_1 PATH_TO_YOUR_YAML_FILE
```
Our code is validated under 2-GPUs setting. `GPU_ID_0` and `GPU_ID_1` are the indices of the selected GPUs. `PATH_TO_YOUR_YAML_FILE` is the path to your config yaml file. We also offer the template of config file `configs/duke2market.yml`, `configs/market2duke.yml`, `configs/single_domain.yml`. You can optionally adjust the hyper-parameters in the config yaml file. All of our experiments are conducted under the mix-precision training to reduce the burden of GPU memory, *i.e*, we set  the flag `fp16=true`.

During the training, the checkpoint files and logs will be saved in `./checkpoints` and `./logs` directories, respectively.

## Test
In our code, the model is evaluated on the target domain at intervals automatically.
You can also evaluate the trained model manually by running:
```shell script
python3 eval.py GPU_ID PATH_TO_CHECKPOINT_FILE [--dataset DATASET]
```

`PATH_TO_CHECKPOINT_FILE` is the path to the checkpoint file of the trained model. `DATASET` is the name of the target dataset. Its value can be `{market,duke,cuhk,msmt}`. As an intermediate product, the feature matrix of target images is stored in the `./features` directory.

## Citation

	@inproceedings{luo2020generalizing,
	  title={Generalizing Person Re-Identification by Camera-Aware Invariance Learning and Cross-Domain Mixup},
	  author={Luo, Chuanchen and Song, Chunfeng and Zhang, Zhaoxiang},
	  booktitle={European Conference on Computer Vision},
	  year={2020}
	}







