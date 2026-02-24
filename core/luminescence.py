import numpy as np

def calculate_cvar(returns: np.ndarray, weights: np.ndarray, confidence_level: float = 0.95) -> float:
    port_returns = returns.dot(weights)
    var_threshold = np.percentile(port_returns, 100 * (1 - confidence_level))
    tail_losses = port_returns[port_returns <= var_threshold]
    return -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

def calculate_max_drawdown(returns: np.ndarray, weights: np.ndarray) -> float:
    port_returns = returns.dot(weights)
    cum_returns = (1 + port_returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    return np.min(drawdowns)

def rmt_eigenvalue_clipping(cov_matrix: np.ndarray, q: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    lambda_max = 1 * (1 + (1/q))**2
    clipped_evals = np.clip(eigenvalues, a_min=lambda_max, a_max=None)
    clipped_cov = eigenvectors @ np.diag(clipped_evals) @ eigenvectors.T
    return clipped_cov

def black_litterman_synthesis(market_returns: np.ndarray, views: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    tau = 0.05
    P = np.eye(len(views))
    omega = np.diag(np.diag(P @ cov_matrix @ P.T)) * tau
    M_inv = np.linalg.inv(tau * cov_matrix)
    term1 = np.linalg.inv(M_inv + P.T @ np.linalg.inv(omega) @ P)
    term2 = M_inv @ market_returns + P.T @ np.linalg.inv(omega) @ views
    return term1 @ term2
