# CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@50-5000 --train_dir ./experiments/fixmatch
# CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 --weight_l_ce=1 --train_dir ./experiments/fixmatch --devicenum=4 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 --weight_l_ce=0 --train_dir ./experiments/fixmatch --devicenum=4
# CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 --weight_l_ce=0 --train_dir ./experiments/fixmatch --devicenum=4

# CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_50.1@100-5000 \
# --weight_l_ce=1 --train_dir ./experiments/fixmatch --devicenum=4

# CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_50.1@100-5000 \
# --weight_l_ce=1 --train_dir ./experiments/fixmatch --devicenum=4 --confidence=0.0

# CUDA_VISIBLE_DEVICES=1 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 --weight_l_ce=0 --train_dir ./experiments/fixmatch --devicenum=1 --use_bn 0

# CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 --weight_l_ce=0 --weight_ulb=1 --train_dir ./experiments/fixmatch --devicenum=4

# CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_50.1@100-5000 \
# --weight_l_ce=1 --weight_ulb=1 --train_dir ./experiments/fixmatch --devicenum=4

# cifar10 50-50
CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@50-5000 \
--weight_l_ce=1 --weight_ulb=1 --train_dir ./experiments/fixmatch --devicenum=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=cifar10_LT_20.1@20-5000 \
--weight_l_ce=4 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4

# cifar10 50-100
CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 \
--weight_l_ce=1 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4
CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 \
--weight_l_ce=0 --weight_ulb=1 --train_dir ./experiments/fixmatch --devicenum=4


CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@100-5000 \
--weight_l_ce=0 --weight_ulb=1 --train_dir ./experiments/fixmatch --devicenum=4


#cifar100 
CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_100.1@100-5000 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4


CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_100.1@100-5000 \
--weight_l_ce=1 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_100.1@100-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4


CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_50.1@50-5000 \
--weight_l_ce=4 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4

CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100.1@400-1 \
--weight_l_ce=4 --weight_ulb=1 --db=2 --train_dir ./experiments/fixmatch --devicenum=4 --upper=5.0



# stl10
CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --scales=4 --wd=0.0005 --dataset=stl10_rand.1@1000-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4 

CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --scales=4 --dataset=stl10_LT_50.1@50-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4

CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --scales=4 --dataset=stl10.1@1000-1 \
--weight_l_ce=4 --weight_ulb=1 --db=2 --upper=5.0  --train_dir ./experiments/fixmatch --devicenum=4


# svhn
CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=svhn_rand.1@1000-5000 \
--weight_l_ce=4 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=svhn_LT_50.1@50-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=svhn_LT_50.1@50-1 \
--weight_l_ce=0 --weight_ulb=2 --train_dir ./experiments/fixmatch --devicenum=4 




# base
CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=cifar10.2@250-1 \
--weight_l_ce=4 --weight_ulb=1 --db=2 --train_dir ./experiments/fixmatch --devicenum=4 --upper=5.0


CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=128 --wd=0.0015 --dataset=cifar100.1@10000-1 \
--weight_l_ce=4 --weight_ulb=1 --db=2 --train_dir ./experiments/fixmatch --devicenum=4 --upper=5.0



# mini-imagenet
CUDA_VISIBLE_DEVICES=2 python fixmatch.py --filters=64 --dataset=tmp.1@10-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=0



# mixmatch
CUDA_VISIBLE_DEVICES=0,1,2,3 python mixmatch.py --filters=32 --dataset=cifar10_LT_50.1@50-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/mixmatch --devicenum=4 --upper=5.0 --w_match=75 --beta=0.75


CUDA_VISIBLE_DEVICES=2,0,1,3 python mixmatch.py --filters=128 --dataset=cifar100_LT_200.1@200-1 \
--weight_l_ce=4 --weight_ulb=0 --train_dir ./experiments/mixmatch --devicenum=4 --upper=5.0 --w_match=75 --beta=0.75


# remixmatch
CUDA_VISIBLE_DEVICES=4,5,6,7 python cta/cta_remixmatch.py --filters=32 --K=4 \
--dataset=cifar10_LT_20.1@20-1 --weight_l_ce=4 --weight_ulb=0  --w_match=1.5 --beta=0.75 --train_dir ./experiments/remixmatch --upper=5.0



#inverse:
CUDA_VISIBLE_DEVICES=0,1,2,3 python fixmatch.py --filters=32 --dataset=cifar10_IT_50.1@40-1 \
--weight_l_ce=0 --weight_ulb=0 --train_dir ./experiments/fixmatch --devicenum=4
