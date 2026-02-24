import torch
import numpy as np
from quark.base import BaseObjective, BaseConstraint

class SwarmTensor:
    def __init__(self, num_fireflies: int = 50, max_gen: int = 150):
        self.nf = num_fireflies
        self.max_gen = max_gen
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.best_weights_ = None
        self.best_fitness_ = None
        self.is_illuminated_ = False

    def _levy_flight(self, d: int, beta: float = 1.5):
        sigma = (torch.lgamma(torch.tensor(1 + beta)) * torch.sin(torch.tensor(torch.pi * beta / 2)) / 
                (torch.lgamma(torch.tensor((1 + beta) / 2)) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = torch.randn(d, device=self.device) * sigma
        v = torch.randn(d, device=self.device)
        return u / torch.abs(v) ** (1 / beta)

    def illuminate(self, objective: BaseObjective, n_assets: int, constraints: BaseConstraint, verbose: bool = False):
        swarm = torch.rand((self.nf, n_assets), dtype=torch.float32, device=self.device)
        for i in range(self.nf):
            swarm[i] = torch.tensor(constraints.map_to_feasible_space(swarm[i].cpu().numpy()), device=self.device)
            
        light = torch.zeros(self.nf, device=self.device)
        for i in range(self.nf):
            light[i] = objective.measure_brightness(swarm[i].cpu().numpy())
            
        self.best_fitness_ = light.min().item()
        self.best_weights_ = swarm[light.argmin()].cpu().numpy()
        
        alpha = 0.5
        for gen in range(self.max_gen):
            for i in range(self.nf):
                for j in range(self.nf):
                    if light[j] < light[i]:
                        r = torch.norm(swarm[i] - swarm[j])
                        attract = 1.0 * torch.exp(-1.0 * r ** 2)
                        levy = self._levy_flight(n_assets)
                        swarm[i] = swarm[i] + attract * (swarm[j] - swarm[i]) + alpha * levy
                swarm[i] = torch.tensor(constraints.map_to_feasible_space(swarm[i].cpu().numpy()), device=self.device)
                
            for i in range(self.nf):
                f = objective.measure_brightness(swarm[i].cpu().numpy())
                light[i] = f
                if f < self.best_fitness_:
                    self.best_fitness_ = f
                    self.best_weights_ = swarm[i].cpu().numpy()
            alpha *= 0.97
        self.is_illuminated_ = True
        return self
