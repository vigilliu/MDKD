
 # MDKD

Besides, some multi-teacher compared approaches such as AVER-MKD, EBKD, AEKD are contained in this repo together.

## Installation
This repo was tested with Python 3.6, PyTorch 1.8.1, and CUDA 11.1.

## Running
1. Train multiple teacher models
``` shell
python train_teacher.py --model resnet32x4 --gpu_id 0 --trial 4
```
After the training is complete, be sure to put the teacher model directory in setting.py.

2. Distill student model
``` shell
python train_student.py --model_s vgg8 --teacher_num 3  --distill inter --ensemble_method CAMKD --nesterov -r 1 -a 1 -b 50 --trial b=50_resnet32x4tovgg8  --gpu_id 1
python train_student.py --model_s MobileNetV2 --teacher_num 3 --distill inter --ensemble_method CAMKD --nesterov -r 1 -a 1 -b 50 --trial final_dkd+klweight_a1b50_Resnet56toMobileNetV2 --gpu_id 1
```
where the flags are explained as:
* `--distill`: specify the distillation method, e.g. `kd`, `hint`
* `--model_s`: specify the student model, see 'models/__init__.py' to check the available model types.
* `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
* `-a`: the weight of the KD loss, default: `1`
* `-b`: the weight of other distillation losses, default: `0`
* `--teacher_num`: specify the ensemble size (number of teacher models)
* `--ensemble_method`: specify the ensemble_method, e.g. `AVERAGE_LOSS`, `AEKD`, `CAMKD`
  
The run scripts for all comparison methods can be found in `scripts/run.sh`.


The implementation of compared methods are mainly based on the author-provided code and the open-source benchmark https://github.com/HobbitLong/RepDistiller. 
