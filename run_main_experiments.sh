#!/usr/bin/env bash

DEVICE="cuda:0"
FIGURES=()

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --figures)
            shift
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                FIGURES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# If no figures specified, run all
RUN_ALL=false
if [ ${#FIGURES[@]} -eq 0 ]; then
    RUN_ALL=true
fi

run_figure() {
    local f=$1
    if $RUN_ALL; then
        return 0
    fi
    for fig in "${FIGURES[@]}"; do
        if [ "$fig" = "$f" ]; then
            return 0
        fi
    done
    return 1
}

#######################################
# FIGURE 1
#######################################
if run_figure 1; then
    python grokking_experiments.py --lr 0.01 --num_epochs 80000 --log_frequency 5000 --device "$DEVICE" --train_fraction 0.4 --loss_function stablemax --beta2 0.999
    python grokking_experiments.py --lr 0.01 --num_epochs 300 --log_frequency 10 --device "$DEVICE" --train_fraction 0.4 --beta2 0.99 --orthogonal_gradients
    python grokking_experiments.py --lr 0.01 --num_epochs 80000 --log_frequency 5000 --device "$DEVICE" --train_fraction 0.4 --softmax_precision 64
    python grokking_experiments.py --lr 0.01 --num_epochs 80000 --log_frequency 5000 --device "$DEVICE" --train_fraction 0.4 --softmax_precision 64 --dropout_rate 0.2
fi

#######################################
# FIGURE 2
#######################################
if run_figure 2; then
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.4 --softmax_precision 16 --adam_epsilon 1e-30
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.4 --softmax_precision 32 --adam_epsilon 1e-30
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.4 --softmax_precision 64 --adam_epsilon 1e-30

    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.6 --softmax_precision 16
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.6 --softmax_precision 32
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.6 --softmax_precision 64

    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.7 --softmax_precision 16
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.7 --softmax_precision 32
    python grokking_experiments.py --lr 0.0005 --num_epochs 20000 --log_frequency 500 --device "$DEVICE" --train_fraction 0.7 --softmax_precision 64
fi

#######################################
# FIGURE 4
#######################################
if run_figure 4; then
    python grokking_experiments.py --lr 0.01 --num_epochs 100000 --log_frequency 5000 --device "$DEVICE" --train_fraction 0.4 --loss_function stablemax --beta2 0.999
    python grokking_experiments.py --lr 0.01 --num_epochs 100000 --log_frequency 5000 --device "$DEVICE" --train_fraction 0.4 --loss_function stablemax --binary_operation product_mod --beta2 0.999
    python grokking_experiments.py --lr 0.01 --num_epochs 100000 --log_frequency 5000 --device "$DEVICE" \
      --train_fraction 0.5 --loss_function stablemax --dataset sparse_parity \
      --num_noise_features 40 --num_parity_features 3 --num_samples 2000 --adam_epsilon 1e-18 --beta2 0.999
fi

#######################################
# FIGURE 6
#######################################
if run_figure 6; then
    python grokking_experiments.py --lr 0.001 --num_epochs 5000 --log_frequency 200 --device "$DEVICE" \
      --train_fraction 0.4 --orthogonal_gradients --use_transformer

    python grokking_experiments.py --lr 0.001 --num_epochs 5000 --log_frequency 200 --device "$DEVICE" \
      --train_fraction 0.4 --use_transformer --weight_decay 1.5

    python grokking_experiments.py --lr 0.001 --num_epochs 5000 --log_frequency 200 --device "$DEVICE" \
      --train_fraction 0.4 --use_transformer

    python grokking_experiments.py --lr 0.005 --num_epochs 500 --log_frequency 20 --device "$DEVICE"\
     --train_fraction 0.4
    python grokking_experiments.py --lr 0.005 --num_epochs 500 --log_frequency 20 --device "$DEVICE"\
     --train_fraction 0.4 --orthogonal_gradients
    python grokking_experiments.py --lr 10 --num_epochs 500 --log_frequency 20 --device "$DEVICE"\
    --train_fraction 0.4 --orthogonal_gradients --optimizer SGD --train_precision 64 --softmax_precision 64\
     --loss_function stablemax
fi