import numpy as np
from .model import predict_linear
from .metrics import mse


def compute_gradients(x, y, w, b): 
    n = len(x)
    y_hat = predict_linear(x, w, b)
    errors = y - y_hat
    dw = (-2.0 / n) * np.sum(x * errors)
    db = (-2.0 / n) * np.sum(errors)
    return dw, db

def gradient_descent(x, y, w0=0.0, b0=0.0, lr=0.03, steps=60):
    w = float(w0)
    b = float(b0)
    history_w = []
    history_b = []
    history_mse = []

    for t in range(steps):
        y_hat = predict_linear(x, w, b)
        current_mse = mse(y, y_hat)
        dw, db = compute_gradients(x, y, w, b)
        w = w - lr * dw
        b = b - lr * db
        history_w.append(w)
        history_b.append(b)
        history_mse.append(current_mse)

    result = {
        "w": w,
        "b": b,
        "history_w": history_w,
        "history_b": history_b,
        "history_mse": history_mse,
    }
    return result
