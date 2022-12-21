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
%run  ./task/experiment_pend_2/train.py \
    --taskname 'pend_2_hnn' \
    --net 'hnn' \
    --iterations 10000 \
    --train_num 90 \
    --test_num 10 \
    --dataset_url 'https://drive.google.com/file/d/1R9PU4JhceGllcEYKSJd45KEXntMVzGIf/view?usp=sharing' \
    --lr 1e-2 \
    --scheduler 'MultiStepLR' \
    --print_every 1

 %run  ./task/analysis/analyze-pend-2/plot_traj.py

 %rm -rf ./training_file