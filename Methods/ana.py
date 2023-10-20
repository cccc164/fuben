import numpy as np
import matplotlib.pyplot as plt

def anaXt(t, X):
    return (1 - t) * X

def anaSigmat(t, sigma0=0.1, sigma1=1):
    return (sigma1 - sigma0) * t + sigma0

def anaScore(x, X, t, sigma0, sigma1):
    return (x - anaXt(t, X)) / anaSigmat(t, sigma0, sigma1)

def exp2(x):
    return np.exp(-0.5 * x ** 2)

def anaW(x, t, Y, sigma0, sigma1, funkernel=exp2):
    W = funkernel(((x[np.repeat(np.arange(x.shape[0]), Y.shape[0]), :] - anaXt(t, Y)[np.repeat(np.arange(Y.shape[0]), x.shape[0]), :]) / anaSigmat(t, sigma0, sigma1)) ** 2 @ np.ones((Y.shape[0], 1)))
    e = W / np.sum(W, axis=1, keepdims=True)
    if np.any(e):
        print("sum((e @ np.ones((1, W.shape[1]))) - 1) ^ 2 < 0.0001 not satisfied")
    return e

def anaV(x, t, Y, sigma0, sigma1):
    W = anaW(x, t, Y, sigma0, sigma1)
    i = np.repeat(np.arange(x.shape[0]), Y.shape[0])
    j = np.repeat(np.arange(Y.shape[0]), x.shape[0])
    Vind = np.reshape((t * W * (-Y[j, :] + ((sigma1 - sigma0) / anaSigmat(t, sigma0, sigma1)) * (x[i, :] - anaXt(t, Y)[j, :]))), (Y.shape[0], -1))
    Vall = np.reshape(np.sum(W * Vind, axis=1), (x.shape[0], -1))
    return Vall


def sphTrans(Y, weights=None, p=None, *args, **kwargs):
    cY = Y.__class__.__name__
    existsIdt = [method.replace("idt.", "") for method in dir(idt) if method.startswith("idt.")]

    if weights is None:
        if "data.frame" in cY:
            b = np.mean(Y, axis=0)
        elif any(ext in cY for ext in existsIdt):
            b = np.mean(idt(Y))
        else:
            b = np.mean(Y, axis=0)
        vY = np.var(idt(Y))
    elif len(weights) == Y.shape[0]:
        aux = cov.wt(idt(Y), wt=weights)
        b = aux.center
        vY = aux.cov
    else:
        raise ValueError("Meaningless weights provided, wrong length?")

    if p is None:
        p = np.arange(Y.shape[1]) + 1
    else:
        p = p[(p <= Y.shape[1])]

    U, d, Vt = svd(vY)
    A = U[:, p - 1] @ np.diag(1 / np.sqrt(d[p - 1])) @ Vt[p - 1, :]
    Ai = U[:, p - 1] @ np.diag(np.sqrt(d[p - 1])) @ Vt[p - 1, :]

    def f(x, inv=False):
        if inv:
            if isinstance(x, np.ndarray):
                out = (Ai @ np.transpose(x) + np.transpose([b])).T
            else:
                out = Ai @ x + b
            return idtInv(out, orig=Y)
        else:
            if isinstance(x, np.ndarray):
                out = np.transpose(A @ (np.transpose(x) - np.transpose([b])))
            else:
                out = A @ (idt(x) - b)
                out.__dict__.pop("orig", None)
                out.__dict__.pop("V", None)
            return out

    return f


def anaForward(x, Y, sigma0, sigma1 = 1 + sigma0, steps=30, plt_bool=False, sphere=True, weights=None):
    if weights is None:
        weights = np.ones(Y.shape[0])
    if weights.shape[0] != Y.shape[0]:
        raise ValueError("weights provided not compatible with the number of rows in Y")
    if sphere:
        st = sphTrans(Y, weights)
    else:
        st = lambda x: x
    x = st(x.T)
    Y = st(Y.T)
    h = 1 / steps
    if plt_bool:
        plt.plot(x[:, 0], x[:, 1], marker='.', linestyle='')
    for i in range(steps):
        v1 = anaV(x, i * h, Y, sigma0, sigma1)
        v2 = anaV(x + h * v1, (i + 1) * h, Y, sigma0, sigma1)
        x = x + 0.5 * h * (v1 + v2)
        if plt_bool:
            plt.plot(x[:, 0], x[:, 1], marker='.', linestyle='')
    return x.T

def anaBackward(x, Y, sigma0, sigma1=1 + sigma0, steps=30, plt_bool=False, sphere=True, weights=None):
    if weights is None:
        weights = np.ones(Y.shape[0])
    if weights.shape[0] != Y.shape[0]:
        raise ValueError("weights provided not compatible with the number of rows in Y")
    if sphere:
        st = sphTrans(Y, weights)
    else:
        st = lambda x: x
    x = st(x.T)
    Y = st(Y.T)
    h = 1 / steps
    if plt_bool:
        plt.plot(x[:, 0], x[:, 1], marker='.', linestyle='')
    for i in range(steps):
        v1 = anaV(x, (steps - i) * h, Y, sigma0, sigma1)
        v2 = anaV(x - h * v1, (steps - i - 1) * h, Y, sigma0, sigma1)
        x = x - 0.5 * h * (v1 + v2)
        if plt_bool:
            plt.plot(x[:, 0], x[:, 1], marker='.', linestyle='')
    return x.T


# Example usage
x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Sample input data
Y = np.array([[1, 1], [2, 2]])  # Sample target data
sigma0 = 0.1  # Initial sigma value

result_forward = anaForward(x, Y, sigma0, plt_bool=True)  # Run the forward pass
result_backward = anaBackward(x, Y, sigma0, plt_bool=True)  # Run the backward pass