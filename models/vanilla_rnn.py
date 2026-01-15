import torch
import torch.nn as nn
from torch.optim import Adam


def build_rnn(input_size, hidden_size, num_layers, bias, batch_first, rnn_type):
    if rnn_type == 'RNN':
        rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='tanh', bias=bias, batch_first=batch_first, dtype=torch.float32)
    elif rnn_type == 'GRU':
        rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dtype=torch.float32)
    else:
        raise ValueError(f'invalid rnn type {rnn_type}')

    return rnn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, rnn_type='GRU'):
        super().__init__()
        self.rnn = build_rnn(input_size, hidden_size, num_layers, bias, batch_first, rnn_type)

    def forward(self, x):
        out, hx = self.rnn(x)
        return out[-1]


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, rnn_type='GRU'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = build_rnn(input_size, hidden_size, num_layers, bias, batch_first, rnn_type)
        self.head = nn.Linear(in_features=hidden_size, out_features=input_size, bias=bias)

    def autoregressive(self, context, length):
        # context = initial hidden state, length = number of autoregressive steps, x = start token
        if len(context.shape) == 1:
            h = torch.broadcast_to(context, (self.num_layers, self.hidden_size))
            x = torch.zeros(1, self.input_size, device=context.device)
        elif len(context.shape) == 2:
            h = torch.broadcast_to(context, (self.num_layers, context.shape[0], self.hidden_size))
            x = torch.zeros(1, context.shape[0], self.input_size, device=context.device)
        else:
            raise ValueError(f'invalid context shape {context.shape}')

        h = h.contiguous()
        x = x.contiguous()

        outputs = []
        for i in range(length):
            x, h = self.rnn(x, h)
            x = self.head(x)
            outputs.append(x)

        return torch.cat(outputs, dim=0)

    def forward(self, context, length):
        return self.autoregressive(context, length)


class RNNAutoencoder(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type, num_layers=1):
        super().__init__()
        self.enc = Encoder(input_size, hidden_size, rnn_type=rnn_type, num_layers=num_layers)
        self.dec = Decoder(input_size, hidden_size, rnn_type=rnn_type, num_layers=num_layers)

    def forward(self, x):
        return self.dec(self.enc(x), x.shape[0])


class Model(RNNAutoencoder):

    def __init__(self, cfg, **kwargs):
        super().__init__(input_size=cfg.enc_in, hidden_size=cfg.d_model, rnn_type=cfg.rnn_type, num_layers=cfg.e_layers)

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


def main():
    print(torch.__version__)

    x = torch.rand(10, 3, 1, dtype=torch.float32)
    # enc = Encoder(1, 4)
    # dec = Decoder(1, 4)
    auto = RNNAutoencoder(1, 4, 'GRU')

    optimizer = Adam(auto.parameters())
    criterion = nn.MSELoss()

    y_pred = auto(x)

    print(list(auto.parameters())[0][0])
    optimizer.zero_grad()
    loss = criterion(y_pred, x)
    loss.backward()
    optimizer.step()
    print(list(auto.parameters())[0][0])

    print(loss.item())


if __name__ == '__main__':
    main()
