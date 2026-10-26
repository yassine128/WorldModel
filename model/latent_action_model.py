import torch
import torch.nn as nn
import torch.nn.functional as F

class LAM(nn.Module):
    def __init__(self, latent_dim=128, d_model=128, n_codes=4):
        super().__init__()
        self.proj = nn.Linear(latent_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 2, d_model))
        self.encoder = self._build_encoder(d_model)
        self.to_code = nn.Linear(d_model, n_codes)
        self.code_emb = nn.Embedding(n_codes, d_model)
        self.decoder = self._build_decoder(latent_dim, d_model)

    def _build_encoder(self, d_model):
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=2)

    def _build_decoder(self, latent_dim, d_model):
        return nn.Sequential(
            nn.Linear(latent_dim + d_model, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, latent_t, latent_t1):
        x = torch.stack((latent_t, latent_t1), dim=1)
        x = self.proj(x) + self.pos
        h = self.encoder(x)
        logits = self.to_code(h.mean(dim=1))
        soft = F.softmax(logits, dim=-1)
        idx = logits.argmax(-1)
        hard = F.one_hot(idx, logits.size(-1)).float()
        onehot = hard + soft - soft.detach()
        code = onehot @ self.code_emb.weight
        latent_t1_pred = self.decoder(torch.cat((latent_t, code), dim=-1))
        return latent_t1_pred, idx