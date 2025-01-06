import torch
import copy
import pandas as pd
import pdb


class MetricsLogger:
    def __init__(self, num_epochs: int, log_frequency: int):
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.num_early_training_epochs = 0
        self.early_training_log_frequency = 10

        if self.num_early_training_epochs == 0:
            self.num_loged_epochs = (num_epochs) // log_frequency
        else:
            self.num_loged_epochs = (num_epochs) // log_frequency 

        # Metrics to log
        self.metric_fns = {
            "loss": self.compute_loss,
            "accuracy": self.compute_accuracy,
            "weights_l2": self.compute_weights_l2,
            "zero_terms": self.compute_zero_terms,
            "softmax_collapse": self.compute_softmax_collapse,
        }

        self.metrics_df = pd.DataFrame(columns=["epoch", "input_type", "metric_name", "layer", "value"])

        self._train_output = None
        self._train_targets = None
        self._test_output = None
        self._test_targets = None

    def _get_epoch_position(self, epoch: int) -> int:
        epoch_position = epoch // self.log_frequency
        if epoch > self.num_early_training_epochs:
            epoch_position += self.num_early_training_epochs // self.early_training_log_frequency
        return epoch_position

    def _run_full_batch_forward(self, model, data, targets, args):
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = model(data)
            if args.use_transformer:
                output = output[:,-1]
        return output, targets

    def log_metrics(self, model, epoch, save_model_checkpoints, saved_models,
                    all_data, all_targets, all_test_data, all_test_targets,
                    args, loss_function):
        """
        Compute and log all metrics in a full-batch setting.
        """

        epoch_position = self._get_epoch_position(epoch)

        if epoch in save_model_checkpoints:
            saved_models[epoch] = copy.deepcopy(model.state_dict())


        self._train_output, self._train_targets = self._run_full_batch_forward(model, all_data, all_targets, args)
        self._test_output, self._test_targets = self._run_full_batch_forward(model, all_test_data, all_test_targets, args)


        for metric_name, metric_fn in self.metric_fns.items():
            rows = metric_fn(model=model,
                             epoch=epoch,
                             epoch_position=epoch_position,
                             args=args,
                             loss_function=loss_function)
            if rows is not None and len(rows) > 0:
                self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame(rows)], ignore_index=True)


        self._train_output = None
        self._train_targets = None
        self._test_output = None
        self._test_targets = None


    def compute_loss(self, model, epoch, epoch_position, args, loss_function):
        train_loss_val = loss_function(self._train_output, self._train_targets).item()
        test_loss_val = loss_function(self._test_output, self._test_targets).item()

        return [
            {
                "epoch": epoch_position,
                "input_type": "train",
                "metric_name": "loss",
                "layer": None,
                "value": train_loss_val
            },
            {
                "epoch": epoch_position,
                "input_type": "test",
                "metric_name": "loss",
                "layer": None,
                "value": test_loss_val
            }
        ]

    def compute_accuracy(self, model, epoch, epoch_position, args, loss_function):
        train_preds = self._train_output.argmax(dim=1)
        test_preds = self._test_output.argmax(dim=1)

        train_acc = (train_preds == self._train_targets).float().mean().item()
        test_acc = (test_preds == self._test_targets).float().mean().item()

        return [
            {
                "epoch": epoch_position,
                "input_type": "train",
                "metric_name": "accuracy",
                "layer": None,
                "value": train_acc
            },
            {
                "epoch": epoch_position,
                "input_type": "test",
                "metric_name": "accuracy",
                "layer": None,
                "value": test_acc
            }
        ]

    def compute_weights_l2(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            val = param.square().sum().sqrt().item()
            results.append({
                "epoch": epoch_position,
                "input_type": "general",
                "metric_name": "weights_l2",
                "layer": name,
                "value": val
            })
        return results

    def compute_zero_terms(self, model, epoch, epoch_position, args, loss_function):
        full_loss = loss_function(self._train_output, self._train_targets, reduction="none")
        zero_val = ((full_loss == 0).sum().item() / (full_loss.shape[0]))
        return [{
            "epoch": epoch_position,
            "input_type": "train",
            "metric_name": "zero_terms",
            "layer": None,
            "value": zero_val
        }]

    def compute_softmax_collapse(self, model, epoch, epoch_position, args, loss_function):
        float_precision = {
            64: torch.float64,
            32: torch.float32,
            16: torch.float16
        }[args.softmax_precision]
        output = self._train_output.to(float_precision)
        output_off = output - output.amax(dim=1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_exp = torch.sum(exp_output, dim=-1, keepdim=True)
        log_softmax = output_off.amax(dim=1, keepdim=True)- torch.log(sum_exp)
        softmax_collapse = (sum_exp==1).float().mean().item()

        return [{
            "epoch": epoch_position,
            "input_type": "train",
            "metric_name": "softmax_collapse",
            "layer": None,
            "value": softmax_collapse
        }]