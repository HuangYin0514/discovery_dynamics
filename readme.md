- [common commod](#common-commod)
- [1. double pend](#1-double-pend)
  - [1.1. hnn](#11-hnn)
  - [1.2. baseline](#12-baseline)
  - [1.3. test](#13-test)
- [2. three body](#2-three-body)
  - [2.1. hnn](#21-hnn)
  - [2.2. baseline](#22-baseline)
  - [2.3. test](#23-test)

---

# common commod

```
rm -rf outputs 
```

---

# 1. double pend

## 1.1. hnn
``` 
%run  ./task/train.py \
    --taskname pend_2_hnn \
    --tasktype pend \
    --net_name hnn \
    --data_name PendulumData \
    --obj 2 \
    --dim 1 \
    --train_num 90 \
    --test_num 10 \
    --lr 1e-2 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 10000 \
    --print_every 1
``` 

``` 
python  ./task/train.py \
    --taskname pend_2_hnn \
    --tasktype pend \
    --net_name hnn \
    --data_name PendulumData \
    --obj 2 \
    --dim 1 \
    --train_num 3 \
    --test_num 2 \
    --lr 1e-2 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 8 \
    --print_every 1
``` 

## 1.2. baseline
```
%run  ./task/train.py \
    --taskname pend_2_baseline \
    --tasktype pend \
    --net_name baseline \
    --data_name PendulumData \
    --obj 2 \
    --dim 1 \
    --train_num 90 \
    --test_num 10 \
    --lr 1e-2 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 10000 \
    --print_every 1
```

```
python  ./task/train.py \
    --taskname pend_2_baseline \
    --tasktype pend \
    --net_name baseline \
    --data_name PendulumData \
    --obj 2 \
    --dim 1 \
    --train_num 2 \
    --test_num 2 \
    --lr 1e-2 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 10 \
    --print_every 1
```

## 1.3. test
```
%run  ./task/analysis/analyze_pend_2/analyze.py \
    --obj 2 \
    --dim 1 \
    --test_num 100 \
    --t0 0 \
    --t_end 30 \
    --h 0.05

%rm -rf ./training_file
```

---

# 2. three body

## 2.1. hnn
```
%run  ./task/train.py \
    --taskname body_3_hnn \
    --tasktype body \
    --net_name hnn \
    --data_name BodyData \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 3000 \
    --print_every 1
```


## 2.2. baseline
```
%run  ./task/train.py \
    --taskname body_3_baseline \
    --tasktype body \
    --net_name baseline \
    --data_name BodyData \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 3000 \
    --print_every 1
```

```
python  ./task/train.py \
    --taskname body_3_baseline \
    --tasktype body \
    --net_name baseline \
    --data_name BodyData \
    --obj 3 \
    --dim 2 \
    --train_num 2 \
    --test_num 2 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 3 \
    --print_every 1
```

## 2.3. test
```
%run  ./task/analysis/analyze_body_3/analyze.py \
    --obj 3 \
    --dim 2 \
    --test_num 100 \
    --t0 0 \
    --t_end 30 \
    --h 0.05

%rm -rf ./training_file
```

```
python  ./task/analysis/analyze_body_3/analyze.py \
    --obj 3 \
    --dim 2 \
    --test_num 2 \
    --t0 0 \
    --t_end 30 \
    --h 0.05

%rm -rf ./training_file
```