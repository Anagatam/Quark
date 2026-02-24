import numpy as np
from quark.base import BaseObjective, BaseConstraint

class QuarkOptimizer:
    def __init__(self, num_fireflies: int = 50, max_gen: int = 150):
        self.nf = num_fireflies
        self.max_gen = max_gen
        self.best_weights_ = None
        self.best_fitness_ = None
        self.is_illuminated_ = False

    def illuminate(self, objective: BaseObjective, n_assets: int, constraints: BaseConstraint, verbose: bool = False):
        swarm = np.random.rand(self.nf, n_assets)
        for i in range(self.nf):
            swarm[i] = constraints.map_to_feasible_space(swarm[i])
            
        light = np.zeros(self.nf)
        for i in range(self.nf):
            light[i] = objective.measure_brightness(swarm[i])
            
        self.best_fitness_ = np.min(light)
        self.best_weights_ = np.copy(swarm[np.argmin(light)])
        alpha = 0.5

        for gen in range(self.max_gen):
            for i in range(self.nf):
                for j in range(self.nf):
                    if light[j] < light[i]:
                        r = np.linalg.norm(swarm[i] - swarm[j])
                        attract = np.exp(-1.0 * r**2)
                        swarm[i] = swarm[i] + attract * (swarm[j] - swarm[i]) + alpha * (np.random.rand(n_assets) - 0.5)
                swarm[i] = constraints.map_to_feasible_space(swarm[i])

            for i in range(self.nf):
                f = objective.measure_brightness(swarm[i])
                light[i] = f
                if f < self.best_fitness_:
                    self.best_fitness_ = f
                    self.best_weights_ = np.copy(swarm[i])
            alpha *= 0.97
        self.is_illuminated_ = True
        return self
