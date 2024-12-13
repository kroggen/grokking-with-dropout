import random
import time
import torch
import torch.nn as nn
import json
import os
import pdb
from constants import FLOAT_PRECISION_MAP
from logger import MetricsLogger
from torch.utils.data import DataLoader
from utils import (evaluate, 
                   cross_entropy_high_precision,
                   stable_cross_entropy,
                   stable_sum,
                   cross_entropy_float32,
                   kahan_cross_entropy,
                   kahan_sum,
                   cross_entropy_low_precision,
                   get_specified_args,
                   get_dataset,
                   get_model,
                   parse_args,
                   get_optimizer,
                   stablemax_cross_entropy)


class OrthoGrad(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, **base_optimizer_args):
        params = list(params)
        defaults = {}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(params, **base_optimizer_args)
        self.param_groups = self.base_optimizer.param_groups

    @staticmethod
    def _orthogonalize_gradients(params):
        for p in params:
            if p.grad is not None:
                w = p.data.view(-1)
                g = p.grad.data.view(-1)

                squared_norm = torch.dot(w, w) + 1e-30
                proj = torch.dot(w, g) / squared_norm
                g_orth = g - proj * w

                norm_g = torch.norm(g, p=2)
                norm_g_orth = torch.norm(g_orth, p=2) + 1e-30
                g_orth_scaled = g_orth * (norm_g / norm_g_orth)

                p.grad.data.copy_(g_orth_scaled.view(p.grad.shape))

    def step(self, closure=None):
        for group in self.param_groups:
            self._orthogonalize_gradients(group['params'])
        return self.base_optimizer.step(closure)


torch.set_num_threads(5) 
random.seed(2)
torch.manual_seed(42)
parser, args = parse_args()

FLOAT_PRECISION = FLOAT_PRECISION_MAP[args.float_precision]
train_precision = FLOAT_PRECISION_MAP[args.train_precision]

device = args.device
print("Using device:", device)

train_dataset, test_dataset = get_dataset(args)
if args.full_batch:
    args.batch_size = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

torch.save(train_dataset, "last_train_loader.pt")
torch.save(test_dataset, "last_test_loader.pt")

args.lr = args.lr/(args.alpha**2)

model = get_model(args)
logger = MetricsLogger(args.num_epochs, args.log_frequency)

base_optimizer = get_optimizer(model, args)


if args.orthogonal_gradients:
    base_optimizer_cls = type(base_optimizer)
    base_state_dict = base_optimizer.state_dict()
    
    optimizer_args = {
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    optimizer = OrthoGrad(model.parameters(), base_optimizer_cls, **optimizer_args)
    optimizer.load_state_dict(base_state_dict)
else:
    optimizer = base_optimizer


print(args.loss_function)
cross_entropy_function = {
    16: cross_entropy_low_precision,
    32: cross_entropy_float32,
    64: cross_entropy_high_precision
}

loss_functions = {
    "label_smoothing": torch.nn.CrossEntropyLoss(label_smoothing=0.2),
    "cross_entropy": cross_entropy_function[args.float_precision],
    "l1": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "stable_ce": stable_cross_entropy,
    "kahan": kahan_cross_entropy,
    "stablemax": stablemax_cross_entropy
}
loss_function = loss_functions[args.loss_function]
save_model_checkpoints = range(0, args.num_epochs, args.log_frequency)
saved_models = {epoch: None for epoch in save_model_checkpoints}

softmax_temperature = 1

if args.full_batch:
    # Load full training data and targets
    all_data = train_dataset.dataset.data[train_dataset.indices].to(device).to(train_precision)
    all_targets = train_dataset.dataset.targets[train_dataset.indices].to(device).long()

    # Load full test data and targets
    all_test_data = test_dataset.dataset.data[test_dataset.indices].to(device).to(train_precision)
    all_test_targets = test_dataset.dataset.targets[test_dataset.indices].to(device).long()
else:
    # If full_batch == False, you would need to handle differently.
    # The user requested full batch scenario, so we assume args.full_batch is True.
    pass

loss = torch.inf
start_time = time.time()
model.to(device)
for epoch in range(args.num_epochs):
    permutation = torch.randperm(all_data.size(0))
    shuffled_data = all_data[permutation]
    shuffled_targets = all_targets[permutation]
    model.train()
    optimizer.zero_grad()
    output = model(shuffled_data) 
    if args.use_transformer:
        output = output[:, -1]
    output = output*args.alpha
    loss = loss_function(output, shuffled_targets)
    loss.backward()
    optimizer.step()


    if epoch % logger.log_frequency == 0:
        logger.log_metrics(
            model=model,
            epoch=epoch,
            save_model_checkpoints=save_model_checkpoints,
            saved_models=saved_models,
            all_data=shuffled_data,
            all_targets=shuffled_targets,
            all_test_data=all_test_data,
            all_test_targets=all_test_targets,
            args=args,
            loss_function=loss_function,
        )

        print(f'Epoch {epoch}: Training loss: {loss.item():.4f}')
        if epoch > 0:
            print(f"Time taken for the last {args.log_frequency} epochs: {(time.time() - start_time)/60:.2f} min")
        start_time = time.time()

model.eval().to('cpu')
test_loss, test_accuracy = evaluate(model, test_loader)
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')
args.lr = args.lr*(args.alpha**2)

specified_args = get_specified_args(parser, args)
if len(specified_args.keys()) == 0:
    experiment_key = f'{args.dataset}_default'
else:
    experiment_key = f'{args.dataset}|' + '|'.join([f'{key}-{str(specified_args[key])}' for key in specified_args.keys()])

torch.save(saved_models, 'last_run_saved_model_checkpoints.pt')
torch.save(optimizer, 'last_optimizer.pt')

os.makedirs(f"loggs/{experiment_key}", exist_ok=True)
logger.metrics_df.to_csv(f"loggs/{experiment_key}/metrics.csv", index=False)

with open(f"loggs/{experiment_key}/args.json", 'w') as f:
    json.dump(vars(args), f, indent=4)

print(f"Saving run: {experiment_key}")