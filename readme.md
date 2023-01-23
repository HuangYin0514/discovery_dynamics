- [1. double pend](#1-double-pend)
    - [1.1. hnn](#11-hnn)
- [2. three body](#2-three-body)
    - [2.1. hnn](#21-hnn)

---

# 1. double pend

## 1.1. hnn

``` 
%run  ./task/train.py \
    --taskname pend_2_task \
    --obj 2 \
    --dim 1 \
    --net_name mechanicsNN \
    --data_name Pendulum2 \
    --train_num 90 \
    --test_num 10 \
    --lr 5e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 10000 \
    --print_every  500 \
    --download_data True

%run  ./task/analyze.py \
    --taskname pend_2_task \
    --net_name mechanicsNN \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```

---

# 2. three body

## 2.1. hnn

```
%run  ./task/train.py \
    --taskname body_3_task \
    --net_name mechanicsNN \
    --data_name Body3 \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 3000 \
    --print_every 150 \
    --download_data True

%run  ./task/analyze.py \
    --taskname body_3_task \
    --net_name mechanicsNN \
    --data_name Body3 \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```