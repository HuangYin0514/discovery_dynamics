# body3 shell

## Baseline_body3

```
%run  ./task/train.py \
    --taskname body_3_task_Baseline \
    --net_name Baseline_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR_body3 \
    --iterations 3000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname body_3_task_Baseline \
    --net_name Baseline_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```

## LNN_body3

```
%run  ./task/train.py \
    --taskname body_3_task_LNN \
    --net_name LNN_body3 \
    --data_name Body3_L \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR_body3 \
    --iterations 3000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname body_3_task_LNN \
    --net_name LNN_body3 \
    --data_name Body3_L \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```

## MechanicsNN_body3

```
%run  ./task/train.py \
    --taskname body_3_task_MechanicsNN \
    --net_name MechanicsNN_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR_body3 \
    --iterations 3000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname body_3_task_MechanicsNN \
    --net_name MechanicsNN_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```

## HNN_body3

```
%run  ./task/train.py \
    --taskname body_3_task_hnn \
    --net_name HNN_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 100 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR_body3 \
    --iterations 3000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname body_3_task_hnn \
    --net_name HNN_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```

## ModLaNet_body3

```
%run  ./task/train.py \
    --taskname body_3_task_ModLaNet \
    --net_name ModLaNet_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR_body3 \
    --iterations 3000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname body_3_task_ModLaNet \
    --net_name ModLaNet_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```

## HnnModScale_body3

```
%run  ./task/train.py \
    --taskname body_3_task_HnnModScale_body3 \
    --net_name HnnModScale_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 \
    --train_num 180 \
    --test_num 20 \
    --lr 1e-3 \
    --criterion L2_norm_loss \
    --optimizer adam \
    --scheduler LambdaLR_body3 \
    --iterations 3000 \
    --print_every 150 \
    --download_data False

%run  ./task/analyze.py \
    --taskname body_3_task_HnnModScale_body3 \
    --net_name HnnModScale_body3 \
    --data_name Body3 \
    --obj 3 \
    --dim 2 

%rm -rf ./training_file
```
