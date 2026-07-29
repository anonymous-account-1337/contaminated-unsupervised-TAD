import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rnn(rnn_type, input_size, hidden_size, num_layers=1, bias=True):
    rnn_type = rnn_type.upper()
    if rnn_type == 'RNN':
        rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='relu', bias=bias, batch_first=True, dtype=torch.float32)
    elif rnn_type == 'GRU':
        rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dtype=torch.float32)
    elif rnn_type == 'LSTM':
        rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dtype=torch.float32)
    else:
        raise ValueError(f'invalid rnn type {rnn_type}')

    return rnn


class Encoder(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size):
        super().__init__()
        self.enc = build_rnn(rnn_type, input_size, hidden_size)

    def forward(self, x):
        if isinstance(self.enc, nn.LSTM):
            _, (h, _) = self.enc(x, hx=None)
        else:
            _, h = self.enc(x, hx=None)
        return h.permute(1, 0, 2)


class Decoder(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size):
        super().__init__()
        self.dec = build_rnn(rnn_type, input_size, hidden_size)
        self.proj = nn.Linear(in_features=hidden_size, out_features=input_size, bias=True)
        self.start_token = nn.Parameter(torch.rand(size=(1, 1, input_size)))

    def forward(self, context, n):
        h = context.permute(1, 0, 2)
        start_token = self.start_token.expand(h.shape[1], 1, 1)
        y = start_token
        c = torch.zeros(size=h.shape).to(device=h.device)

        outputs = []
        for i in range(n):
            if isinstance(self.dec, nn.LSTM):
                _, (h, c) = self.dec(y, hx=(h, c))
            else:
                _, h = self.dec(y, hx=h)
            y = self.proj(h.permute(1, 0, 2))
            outputs.append(y)

        return torch.cat(outputs, dim=1)


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(rnn_type=cfg.rnn_type, input_size=cfg.enc_in, hidden_size=cfg.latent_size)
        self.dec = Decoder(rnn_type=cfg.rnn_type, input_size=cfg.enc_in, hidden_size=cfg.latent_size)

    def forward(self, x):
        return self.dec(self.enc(x), x.shape[1])

    def train_step(self, x, epoch):
        x_reconstructed = self(x)
        return F.mse_loss(x_reconstructed, x)

    def anomaly_score(self, x):
        x_reconstructed = self(x)
        return F.mse_loss(x_reconstructed, x, reduction='none')


def main():
    torch.manual_seed(47)
    hidden_size = 2
    rnn_type = 'rnn'
    x = torch.rand(size=(4, 3, 1))
    enc = Encoder(rnn_type, x.shape[-1], hidden_size)
    dec = Decoder(rnn_type, x.shape[-1], hidden_size)

    print(x.shape)
    h = enc(x)
    print(h.shape)
    y = dec(h, x.shape[1])
    print(y.shape)


if __name__ == '__main__':
    main()
