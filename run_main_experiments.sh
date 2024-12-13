
python grokking_experiments.py --lr 0.01 --num_epochs 80000 --log_frequency 5000 --device cuda:0 --train_fraction 0.4 --loss_function stablemax
python grokking_experiments.py --lr 0.01 --num_epochs 300 --log_frequency 10 --device cuda:0 --train_fraction 0.4
python grokking_experiments.py --lr 0.01 --num_epochs 80000 --log_frequency 5000 --device cuda:0 --train_fraction 0.4


python grokking_experiments.py --lr 0.008 --num_epochs 5000 --log_frequency 200 --device cuda:0 --train_fraction 0.4 --orthogonal_gradients --use_transformer
