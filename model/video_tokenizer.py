import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class vidTokenizer(nn.Module): 
    def __init__(self, learning_rate):
        super().__init__()

        self.latent_dim = 128
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, self.latent_dim),
        )

    def _build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),   
            nn.Sigmoid(),
        )

    def forward(self, X):
        z = self.encoder(X)
        y_pred = self.decoder(z)
        return y_pred

    def fit(self, X, epochs=100, batch_size=32):
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        self.history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for (images,) in loader:
                self.optimizer.zero_grad()

                prediction = self.forward(images)
                loss = self.criterion(prediction, images)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)

            avg_loss = total_loss / len(dataset)
            self.history.append(avg_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {avg_loss:.6f}"
            )

        return self.history

