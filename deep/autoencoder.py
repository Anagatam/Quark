import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class RiskAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_features)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

class DeepCovarianceEstimator:
    def __init__(self, latent_dim: int = 5, epochs: int = 200, lr: float = 0.01):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.is_illuminated_ = False
        print(f"[DEEP LEARNING] Target Device resolved to: {self.device}")
        
    def illuminate(self, returns: np.ndarray):
        X = torch.tensor(returns, dtype=torch.float32, device=self.device)
        self.model = RiskAutoencoder(returns.shape[1], self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        for _ in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(X)
            loss = criterion(recon, X)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            self.reconstructed_returns_ = self.model(X).cpu().numpy()
            self.expected_returns_ = np.mean(self.reconstructed_returns_, axis=0)
            
        residuals = returns - self.reconstructed_returns_
        factor_cov = np.cov(self.reconstructed_returns_, rowvar=False)
        idiosyncratic_var = np.var(residuals, axis=0)
        self.cov_matrix_ = factor_cov + np.diag(idiosyncratic_var)
        self.is_illuminated_ = True
        return self

    def transform(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_illuminated_:
            raise ValueError("Model not illuminated.")
        return self.reconstructed_returns_, self.expected_returns_, self.cov_matrix_
