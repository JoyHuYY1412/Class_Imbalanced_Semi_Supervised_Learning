# CADR_FiXMatch

Code for the paper: "[On Non-Random Missing Labels in Semi-Supervised Learning]()" by 
Xinting Hu, Yulei Niu, Chunyan Miao, Xian-Sheng Hua, Hanwang Zhang

The code is based on [Fixmatch](https://github.com/google-research/fixmatch) by David Berthelot. Thank you for your sharing!


## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Install datasets
The datasets used in this repository are: CIFAR, STL10, and miniImageNet. 
CIFAR and STL10 will be downloaded and converted automatically. For mini-ImageNet, you can download the mini-ImageNet dataset, and convert it to TFrecord use [this](https://github.com/kmonachopoulos/ImageNet-to-TFrecord/tree/c722ad22f72bb8c7c674972b25c35d1734481537). The download link for my converted version is [here](https://drive.google.com/drive/folders/15TBbksuEWYmvN9N40MdFjRxuw96W-Vh2?usp=sharing).

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py


# Create unlabeled datasets 
# unlabeled -- original balanced version
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &

# unlabeled -- Long-Tailed (LT) version 

# unlabeled -- cifar10_LT
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10_LT_20 $ML_DATA/cifar10-train.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10_LT_50 $ML_DATA/cifar10-train.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10_LT_100 $ML_DATA/cifar10_LT_100-train.tfrecord &

# unlabeled -- cifar100_LT
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100_LT_50 $ML_DATA/cifar100_LT_50-train.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100_LT_100 $ML_DATA/cifar100_LT_100-train.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100_LT_200 $ML_DATA/cifar100-train.tfrecord &

# unlabeled -- stl10_LT
python scripts/create_unlabeled.py $ML_DATA/SSL2/stl10_LT_50 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
python scripts/create_unlabeled.py $ML_DATA/SSL2/stl10_LT_100 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &

# unlabeled -- miniImageNet_LT
python scripts/create_unlabeled.py $ML_DATA/SSL2/miniImageNet_LT_100 $ML_DATA/miniImageNet-train.tfrecord 
wait

# Create original semi-supervised subsets (seed for random seed, size for the whole size of the labeled data)
for seed in 1; do
    for size in 40 250 4000; do
        python scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 2500 10000; do
        python scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    python scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done

# for LT-label semi-supervised subsets  (seed for random seed, size for the max size of labeled data among classes, lamda for imabalance ratio of the labeled data)
for seed in 1; do
    for size in 20 50 100; do
        python scripts/create_split.py --seed=$seed --size=$size --lamda=$size $ML_DATA/SSL2/cifar10_LT_$size $ML_DATA/cifar10-train.tfrecord &
    done
done 

for seed in 1; do
    for size in 50 100 200; do
        python scripts/create_split.py --seed=$seed --size=$size --lamda=50 $ML_DATA/SSL2/cifar100_LT_$size $ML_DATA/cifar100-train.tfrecord &
    done
done 

python scripts/create_split.py --seed=1 --size=100 --lamda=100 $ML_DATA/SSL2/miniImageNet_LT_100 $ML_DATA/miniImageNet_LT_100-train.tfrecord &


```
Default available labeled sizes are 10, 20, 30, 40, 100, 250, 1000, 4000.
Default validation available sizes are 1, 5000.
Default possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).
You can change the above default settings in `libml\data.py`.

## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"
```
### Backbone 
We have WideResNet and ResNet18 for backbones, you can choose by modifying `libml/model.py`.

### Example

**For original semi-supervised subsets:**
For example, training a FixMatch with 32 filters on cifar10 shuffled with `seed=1`, 40 labeled samples and 1 validation sample:

Baseline FixMatch:
```bash
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=32 --dataset=cifar10.1@40-1 --train_dir ./experiments/fixmatch
```

Ours:
```bash
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=32 --CAP --CAI --CADR --dataset=cifar10.1@40-1 --train_dir ./experiments/fixmatch
```


**For LT-labeled semi-supervised subsets:**
Baseline FixMatch:
```bash
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=32 --dataset=cifar10_LT_20.1@20-1 --train_dir ./experiments/fixmatch
```

Ours:
```bash
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=32 --CAP --CAI --CADR --dataset=cifar10_LT_20.1@20-1 --train_dir ./experiments/fixmatch
```


#### Multi-GPU training
Just pass more GPUs and fixmatch automatically scales to them, here we assign GPUs 4-7 to the program:
Baseline FixMatch:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=cifar10_LT_20.1@20-1 --train_dir ./experiments/fixmatch --devicenum=4
```

Ours:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --CAP --CAI --CADR --dataset=cifar10_LT_20.1@20-1 --train_dir ./experiments/fixmatch --devicenum=4
```

See `run.sh` for running scripts.


#### Flags

```bash
python fixmatch.py --help
# The following option might be too slow to be really practical.
# python fixmatch.py --helpfull
# So instead I use this hack to find the flags:
fgrep -R flags.DEFINE libml fixmatch.py
```

The `--augment` flag can use a little more explanation. It is composed of 3 values, for example `d.d.d`
(`d`=default augmentation, for example shift/mirror, `x`=identity, e.g. no augmentation, `ra`=rand-augment,
 `rac`=rand-augment + cutout):
- the first `d` refers to data augmentation to apply to the labeled example. 
- the second `d` refers to data augmentation to apply to the weakly augmented unlabeled example. 
- the third `d` refers to data augmentation to apply to the strongly augmented unlabeled example. For the strong
augmentation, `d` is followed by `CTAugment` for `fixmatch.py` and code inside `cta/` folder.



## Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir ./experiments
```

## Checkpoint accuracy

We compute the arithmetic mean accuracy and geometric mean accuracy of the last 10 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.1@40-1, extracting accuracy:
./scripts/extract_accuracy.py ./experiments/fixmatch/cifar10.1@40-1 /CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_filters32_lr0.03_nclass10_repeat4_scales3_uratio7_wd0.0005_wu1.0/

./scripts/extract_gm_accuracy.py ./experiments/fixmatch/cifar10.1@40-1 /CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_filters32_lr0.03_nclass10_repeat4_scales3_uratio7_wd0.0005_wu1.0/

# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## Adding datasets
You can add custom datasets into the codebase by taking the following steps:

1. Add a function to acquire the dataset to `scripts/create_datasets.py` similar to the present ones, e.g. `_load_cifar10`. 
You need to call `_encode_png` to create encoded strings from the original images.
The created function should return a dictionary of the format 
`{'train' : {'images': <encoded 4D NHWC>, 'labels': <1D int array>},
'test' : {'images': <encoded 4D NHWC>, 'labels': <1D int array>}}` .
2. Add the dataset to the variable `CONFIGS` in `scripts/create_datasets.py` with the previous function as loader. 
You can now run the `create_datasets` script to obtain a tf record for it.
3. Use the `create_unlabeled` and `create_split` script to create unlabeled and differently split tf records as above in the *Install Datasets* section.
4. In `libml/data.py` add your dataset in the `create_datasets` function. The specified "label" for the dataset has to match
the created splits for your dataset. You will need to specify the corresponding variables if your dataset 
has a different # of classes than 10 and different resolution and # of channels than 32x32x3
5. In `libml/augment.py` add your dataset to the `DEFAULT_AUGMENT` variable. Primitives "s", "m", "ms" represent mirror, shift and mirror+shift. 

## Citing this work

<!-- ```bibtex
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
 -->