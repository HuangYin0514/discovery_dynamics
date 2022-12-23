########################################################
conda activate pytorch
cd C:\Users\qxcc-qdu\PycharmProjects\discovery_dynamics\baseline
jupyter notebook

#-----------------------
rm  .\experiment-pend-2\data\

jt -t  solarizedl
jt -t solarizedl -fs 11 -cellw 65% -ofs 11 -dfs 11 -T -N

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

 %run  ./task/analysis/analyze-pend-2/plot_traj.py

 %rm -rf ./training_file