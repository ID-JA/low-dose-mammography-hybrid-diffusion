import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        z: [B, C, H, W]
        returns:
            z_q: quantized tensor
            vq_loss: quantization loss
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [BHW, C]

        # distances to embedding vectors
        dist = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # [BHW, K]

        indices = torch.argmin(dist, dim=1)  # [BHW]
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # VQ losses
        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden=64, z_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, z_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden=64, z_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=64, z_channels=64, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, z_channels)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=z_channels)
        self.decoder = Decoder(in_channels, hidden, z_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss
