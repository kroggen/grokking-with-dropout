import argparse
import copy
import itertools
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split
from collections import defaultdict

from binary_operations import (product_mod,
                               add_mod,
                               subtract_mod,
                               factorial,
                               random_map)
from datasets import AlgorithmicDatasetTransformer
from transformers import Transformer, Config
from utils import (evaluate_transformer, 
                   cross_entropy_high_precision, 
                   update_results,
                   stable_sum,
                   parse_args,
                   get_specified_args,
                   stablemax_cross_entropy,
                   cross_entropy_float32)


def split_dataset(dataset, train_fraction, batch_size):
    total_size = len(dataset.data)
    train_size = int(train_fraction * total_size)
    test_size = total_size - train_size
    print(f'Starting trining. Train dataset size: {train_size}, Test size: {test_size}')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #test_dataset.indices = test_dataset.indices[:4000]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    return train_loader, test_loader

def reduce_train_dataset(original_train_dataset, reduced_fraction, batch_size):
    original_indices = original_train_dataset.indices
    reduced_train_size = int(reduced_fraction * len(original_indices))
    reduced_indices = original_indices[:reduced_train_size]
    reduced_train_dataset = Subset(dataset, reduced_indices)
    
    reduced_train_loader = DataLoader(reduced_train_dataset, batch_size=batch_size, shuffle=True)
    return reduced_train_loader

random.seed(2)
torch.manual_seed(2)
parser, args = parse_args()

BINARY_OPERATION = subtract_mod

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dataset = AlgorithmicDatasetTransformer(BINARY_OPERATION, p=args.modulo, input_size=args.input_size, output_size=args.modulo)
args.batch_size = int(len(dataset.data)*args.train_fraction)
train_loader, test_loader = split_dataset(dataset, args.train_fraction, args.batch_size)
torch.save(train_loader, "last_train_loader.pt")
torch.save(test_loader, "last_test_loader.pt")


print("Using AlgorithmicDataset")
config = Config()
model = Transformer(config).to(torch.float32)

initial_model_state = copy.deepcopy(model.state_dict())
if args.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0, eps=1e-8)#, weight_decay=WEIGHT_DECAY)
elif args.optimizer == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-12, weight_decay=args.weight_decay)
elif args.optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
else: 
    raise ValueError(f'Unsupported optimizer type: {args.optimizer}')

loss_functions = {
    "cross_entropy": cross_entropy_float32 if args.float_precision == 32 else cross_entropy_high_precision,
    "l1": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "stablemax": stablemax_cross_entropy
}
loss_function = loss_functions[args.loss_function]

saved_models = {}
metrics = {}
softmax_temperature = 1

if args.full_batch == True:
    
    all_data = dataset.data[train_loader.dataset.indices].to(device).to(torch.int64)
    all_targets = dataset.targets[train_loader.dataset.indices].to(device).to(torch.float32)

    print(all_data.shape)
    print(all_targets.shape)
metrics["normalized_margin"] = []
metrics["loss_after_update"] = []
metrics["zero_terms"] = []
metrics["softmax_collapse"] = []
metrics["exponential_underflow"] = []
metrics["percentage_absoption"] = []
metrics["percentage_zero_grad"] = []
metrics["samples_with_zero_gradients"] = []
metrics["gradient_norm"] = {name:[] for name,_ in model.named_parameters()}
metrics["first_order_moment"] = {i:[] for i in range(3)}
metrics["second_order_moment"] = {i:[] for i in range(3)}
metrics["logit_max"] = []
metrics["logit_min"] = []
metrics["weight_norm"] = {name:[] for name,_ in model.named_parameters()}
metrics["cosine_nlm"] = {name:[] for name,_ in model.named_parameters()}




for epoch in range(args.num_epochs):
    model.train().to(device)
    max_spurious = 0
    if args.temperature_schedule and epoch%10_000==0 and epoch>0:
        softmax_temperature *= 2

    if args.full_batch == True:
        optimizer.zero_grad()
        output = model(all_data)[:,-1]
        
        loss = loss_function(output, all_targets.float())
        loss.backward()
        if epoch % args.logg_frequency  == 0:
            for name, p in model.named_parameters():
                metrics["cosine_nlm"][name].append(torch.nn.functional.cosine_similarity(p.view(1,-1),-p.grad.view(1, -1), dim=1).item())

        if args.orthogonal_gradients:
            for name, param in model.named_parameters():
                if True:
                    if param.grad is not None:
                        w = param.data.view(-1)
                        g = param.grad.data.view(-1)

                        squared_norm = torch.dot(w, w) + 1e-18
                        proj = torch.dot(w, g) / squared_norm
                        
                        g_orth = g - proj * w

                        g_orth = g_orth.view(param.grad.data.shape)
                        norm_g = torch.norm(g)
                        norm_g_orth = torch.norm(g_orth) + 1e-18

                        g_orth_scaled = g_orth * (norm_g / norm_g_orth)
                        param.grad.data.copy_(g_orth_scaled)
        optimizer.step()

    
    if epoch % args.logg_frequency  == 0:
        full_loss = loss_function(output, all_targets, reduction="none")
        metrics["zero_terms"].append(((full_loss==0).sum()/(full_loss.shape[0]*full_loss.shape[1])).item())
        if args.float_precision == 64:
            output = output.to(torch.float64)
        output_off = output- output.amax(1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_function = stable_sum if args.loss_function == "stable_softmax" else torch.sum
        sum_exp = sum_function(exp_output, dim=-1, keepdim=True)
        sofmax_collapse = exp_output.amax(1)==sum_exp.unsqueeze(1)
        exponential_underflow = (sum_exp==0)
        for layer, parameters in model.named_parameters():
            metrics["weight_norm"][layer].append(parameters.square().sum().sqrt().item())

        metrics["logit_max"].append(output.amax(1).mean())
        metrics["logit_min"].append(output.amin(1).mean())

        metrics["softmax_collapse"].append(sofmax_collapse.float().mean().item()) 
        metrics["exponential_underflow"].append(exponential_underflow.float().mean().item()) 
        epoch_metrics = defaultdict(dict)
        epoch_metrics["train_loss"], epoch_metrics["train_accuracy"]= evaluate_transformer(model, train_loader)
        epoch_metrics["test_loss"], epoch_metrics["test_accuracy"] = evaluate_transformer(model, test_loader)
        saved_models[epoch] = copy.deepcopy(model).cpu()
        metrics[epoch] = epoch_metrics
        print(f'Epoch {epoch}: Training loss: {epoch_metrics["train_loss"]:.4f}')
        if epoch>1:
            print(f"Time taken for the last {args.logg_frequency} epochs: {(time.time() - start_time)} seconds")
        start_time = time.time()


model.eval().to('cpu')
test_loss, test_accuracy = evaluate_transformer(model, test_loader, loss_function)
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')
specified_args = get_specified_args(parser, args)
if len(specified_args.keys()) == 0:
    experiment_key = f'{args.dataset}_default'
else:
    experiment_key = f'{args.dataset}|'+ '|'.join([f'{key}-{str(specified_args[key])}' for key in specified_args.keys()])

torch.save(model, f'models/{experiment_key}.pt')
torch.save(saved_models, 'last_run_saved_model_checkpoints.pt')
update_results('experiment_results.pt', experiment_key, metrics)
print(f"Saving run: {experiment_key}")