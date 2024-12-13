import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activations = []
        self.activations_from_abs_input = None
        self.layers = nn.ModuleList()
        self.non_linearity = nn.ReLU()
        self.uses_bias = bias
        self.alpha = 1
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(hidden_sizes)+1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias))
    
    def forward(self, x, keep_activations=False):
        x = x.flatten(start_dim=1)
        if keep_activations:
            self.activations = []
        for i, layer in enumerate(self.layers):
            x = self.non_linearity(layer(x)) if i<len(self.layers) -1 else layer(x)
            if keep_activations and i<len(self.layers):
                self.activations.append(x)
        return x*self.alpha
    

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x_emb = self.embedding(x.long())
        x_emb = x_emb.transpose(0, 1)
        out = self.transformer(x_emb)
        out = self.linear(out.transpose(0, 1))
        return out
    
