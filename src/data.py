import numpy as np

def make_synthetic_data(n=40, x_min=-3.0, x_max=3.0, true_w=2.5, true_b=-1.0, noise_std=1.0, seed=42):
    np.random.seed(seed) 
    x = np.linspace(x_min, x_max, n) 
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n) 
    y = true_w * x + true_b + noise 
    return x, y 