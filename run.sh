########################################################

# 双摆
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

 %run  ./task/analysis/analyze_pend_2/plot_trajectory.py \
    --test_num 1 \
    --t0 0 \
    --t_end 10 \
    --h 0.02

%run  ./task/analysis/analyze_pend_2/plot_bar_error.py  \
    --test_num 100 \
    --t0 0 \
    --t_end 30 \
    --h 0.05

%rm -rf ./training_file

########################################################

# %%time

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

%run  ./task/analysis/analyze_body_3/analyze.py \
    --obj 3 \
    --dim 2 \
    --test_num 100 \
    --t0 0 \
    --t_end 30 \
    --h 0.05

%rm -rf ./training_file