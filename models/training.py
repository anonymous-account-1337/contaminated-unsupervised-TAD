import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Trainer:

    def __init__(self, model: nn.Module):
        self.model = model

    def train_step(self, x, epoch):
        raise NotImplemented

    def validation_step(self, x, epoch):
        raise NotImplemented

    def anomaly_score(self, x):
        raise NotImplemented


class ReconstructionTrainer(Trainer):

    def __init__(self, model):
        super().__init__(model)
        self.optimizer = Adam(self.model.parameters())

    def train_step(self, x, epoch):
        self.optimizer.zero_grad()
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x, reduction='mean')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, x, epoch):
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x, reduction='mean')
        return loss.item()

    def guided_reconstruction(self, x_anom, x_clean):
        self.optimizer.zero_grad()
        latent_anom = self.model.encode(x_anom)
        rec_x_anom = self.model.decode(latent_anom)

        latent_clean = self.model.encode(x_clean)
        rec_x_clean = self.model.decode(latent_clean)

        l1 = F.mse_loss(rec_x_anom, x_clean, reduction='mean') + F.mse_loss(rec_x_clean, x_clean, reduction='mean')
        l2 = F.mse_loss(latent_anom, latent_clean, reduction='mean')
        loss = l1 + l2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def anomaly_score(self, x):
        return F.mse_loss(self.model(x), x, reduction='none')

    def reconstruct(self, x):
        return self.model(x)


class TranADTrainer(Trainer):

    def __init__(self, model):
        super().__init__(model)
        self.optimizer = Adam(self.model.parameters())

    @staticmethod
    def get_alpha(epoch):
        return 1.05 ** (-epoch)

    def train_step(self, x, epoch):
        self.optimizer.zero_grad()
        alpha = self.get_alpha(epoch)
        o1, o2, o2_hat = self.model(x)

        l1 = alpha * F.mse_loss(o1, x, reduction='mean') + (1 - alpha) * F.mse_loss(o2_hat, x, reduction='mean')
        l2 = alpha * F.mse_loss(o2, x, reduction='mean') - (1 - alpha) * F.mse_loss(o2_hat, x, reduction='mean')

        loss = l1 + l2
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, x, epoch):
        alpha = self.get_alpha(epoch)
        o1, o2, o2_hat = self.model(x)

        l1 = alpha * F.mse_loss(o1, x, reduction='mean') + (1 - alpha) * F.mse_loss(o2_hat, x, reduction='mean')
        l2 = alpha * F.mse_loss(o2, x, reduction='mean') - (1 - alpha) * F.mse_loss(o2_hat, x, reduction='mean')

        loss = l1 + l2

        return loss.item()

    def anomaly_score(self, x, alpha=0.5):
        o1, o2, o2_hat = self.model(x)
        return alpha * F.mse_loss(o1, x, reduction='none') + (1 - alpha) * F.mse_loss(o2_hat, x, reduction='none')


class USADTrainer(Trainer):

    def __init__(self, model):
        super().__init__(model)
        self.optimizer1 = Adam(list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()), lr=1e-3)
        self.optimizer2 = Adam(list(self.model.decoder2.parameters()), lr=1e-4)

    @staticmethod
    def get_alpha(epoch):
        return 1 / 1.02 ** epoch

    def train_step(self, x, epoch):
        self.optimizer1.zero_grad()
        w1, w2 = self.model(x)
        _, w21 = self.model(w1)
        alpha = self.get_alpha(epoch)

        l1 = alpha * F.mse_loss(x, w1, reduction='mean') + (1 - alpha) * torch.tanh(F.mse_loss(x, w21, reduction='mean'))
        l1.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        w1, w2 = self.model(x)
        _, w21 = self.model(w1.detach())
        l2 = alpha * F.mse_loss(x, w2, reduction='mean') - (1 - alpha) * torch.tanh(F.mse_loss(x, w21, reduction='mean'))
        l2.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer2.step()
        return l1.item(), l2.item()

    def validation_step(self, x, epoch):
        w1, w2 = self.model(x)
        _, w21 = self.model(w1)
        alpha = self.get_alpha(epoch)

        l1 = alpha * F.mse_loss(x, w1, reduction='mean') + (1 - alpha) * torch.tanh(F.mse_loss(x, w21, reduction='mean'))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        w1, w2 = self.model(x)
        _, w21 = self.model(w1.detach())
        l2 = alpha * F.mse_loss(x, w2, reduction='mean') - (1 - alpha) * torch.tanh(F.mse_loss(x, w21, reduction='mean'))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return l1.item(), l2.item()

    def anomaly_score(self, x, alpha=0.5):
        w1, _ = self.model(x)
        _, w21 = self.model(w1)
        score = alpha * F.mse_loss(x, w1, reduction='none') + (1 - alpha) * F.mse_loss(x, w21, reduction='none')
        return score
