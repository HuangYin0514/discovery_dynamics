# pend2 shell

## Pend2_analytical

```
%run  ./task/train.py \
    --taskname pend_2_task_analytical \
    --net_name Pend2_analytical \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 \
    --lr 1e-3 \
    --criterion L2_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 3 \
    --print_every 1 \

%run  ./task/analyze.py \
    --taskname pend_2_task_analytical \
    --net_name Pend2_analytical \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```

## Baseline_pend2

```
%run  ./task/train.py \
    --taskname pend_2_task_Baseline \
    --net_name Baseline_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 \
    --lr 1e-3 \
    --criterion L2_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 3000 \
    --print_every 1000 \

%run  ./task/analyze.py \
    --taskname pend_2_task_Baseline \
    --net_name Baseline_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file

```

## HNN_pend2

```
# the best choose is 1e-2 and l2_loss
%run  ./task/train.py \
    --taskname pend_2_task_HNN \
    --net_name HNN_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 \
    --lr 1e-2 \
    --criterion L2_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 10000 \
    --print_every 1000 \

%run  ./task/analyze.py \
    --taskname pend_2_task_HNN \
    --net_name HNN_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```

## MechanicsNN_pend2

```
%run  ./task/train.py \
    --taskname pend_2_task_MechanicsNN_pend2 \
    --net_name MechanicsNN_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 \
    --lr 1e-2 \
    --criterion L2_loss \
    --optimizer adam \
    --scheduler MultiStepLR \
    --iterations 10000 \
    --print_every 1000 \

%run  ./task/analyze.py \
    --taskname pend_2_task_MechanicsNN_pend2 \
    --net_name MechanicsNN_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```

## ModLaNet_pend2

```
%run  ./task/train.py \
    --taskname pend_2_task_ModLaNet_pend2 \
    --net_name ModLaNet_pend2 \
    --data_name Pendulum2_L \
    --obj 2 \
    --dim 1 \
    --lr 1e-2 \
    --criterion L2_loss \
    --optimizer adam \
    --scheduler LambdaLR \
    --iterations 10000 \
    --print_every 1000 \
    --download_data False

%run  ./task/analyze.py \
    --taskname pend_2_task_ModLaNet_pend2 \
    --net_name ModLaNet_pend2 \
    --data_name Pendulum2_L \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```

## LNN_pend2

```
%run  ./task/train.py \
    --taskname pend_2_task_LNN_pend2 \
    --net_name LNN_pend2 \
    --data_name Pendulum2_L \
    --obj 2 \
    --dim 1 \
    --train_num 90 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR \
    --iterations 10000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname pend_2_task_LNN_pend2 \
    --net_name LNN_pend2 \
    --data_name Pendulum2_L \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```

## HnnModScale_pend2

```
%run  ./task/train.py \
    --taskname pend_2_task_HnnModScale_pend2 \
    --net_name HnnModScale_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 \
    --lr 1e-2 \
    --criterion L2_loss \
    --optimizer adam \
    --scheduler LambdaLR \
    --iterations 10000 \
    --print_every 1000 \

%run  ./task/analyze.py \
    --taskname pend_2_task_HnnModScale_pend2 \
    --net_name HnnModScale_pend2 \
    --data_name Pendulum2 \
    --obj 2 \
    --dim 1 

%rm -rf ./training_file
```