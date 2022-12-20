########################################################
conda activate pytorch
cd C:\Users\qxcc-qdu\PycharmProjects\discovery_dynamics\baseline
jupyter notebook

#-----------------------
rm  .\experiment-pend-2\data\

jt -t  solarizedl
jt -t solarizedl -fs 11 -cellw 65% -ofs 11 -dfs 11 -T -N

########################################################
# 单摆
python .\experiment-pend-1\train.py --model=hnn --hidden_dim=200 --print_every=500 --plot  --overwrite
python .\experiment-pend-1\train.py --model=hnn --hidden_dim=75  --print_every=500 --plot  --overwrite
python .\experiment-pend-1\train.py --model=lnn --hidden_dim=200 --print_every=500 --plot  --overwrite
python .\experiment-pend-1\train.py --model=lnn --hidden_dim=75 --print_every=500 --plot  --overwrite

# 双摆
%run  ./experiment_pend_2/train.py --model 'baseline' --learn_rate 1e-3 --end_epoch 3000 --print_every 500 --overwrite --verbose --plot
%run  ./experiment_pend_2/train.py --model 'hnn' --hidden_dim 200 --end_epoch 10000 --print_every 500 --plot  --overwrite
%run  ./experiment_pend_2/train.py --model 'hnn' --hidden_dim 75  --end_epoch 10000 --print_every 500 --plot  --overwrite
%run  ./experiment_pend_2/train.py --model 'lnn' --hidden_dim 200 --end_epoch 10000 --print_every 500 --plot  --overwrite
%run  ./experiment_pend_2/train.py --model 'lnn' --hidden_dim 75  --end_epoch 10000 --print_every 500 --plot  --overwrite

%run analysis/analyze-pend-2.py --samples 100 --re_test


# 三体
python ./experiment-body-3/train.py --model 'baseline'  --learn_rate 1e-3 --end_epoch 3000 --overwrite --verbose --plot
python ./experiment-body-3/train.py --model 'hnn'       --learn_rate 1e-3 --end_epoch 3000 --overwrite --verbose --plot

