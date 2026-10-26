import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=128, d_model=256, n_codes=4, n_layers=4, n_heads=4, context=8):
        super().__init__()
        self.context = context
        self.proj = nn.Linear(latent_dim, d_model)
        self.action_emb = nn.Embedding(n_codes, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        self.transformer = self._build_transformer(d_model, n_layers, n_heads)
        self.head = nn.Linear(d_model, latent_dim)

    def _build_transformer(self, d_model, n_layers, n_heads):
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, latents, codes):
        T = latents.size(1)
        x = self.proj(latents) + self.action_emb(codes) + self.pos[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=latents.device)
        h = self.transformer(x, mask=mask)
        return latents + self.head(h)