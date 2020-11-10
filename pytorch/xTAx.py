# Compute the Hessian of x^TAx w.r.t. x

import torch
import time

num_iter = 10
N = 1000

x = torch.rand(N, dtype=torch.float32, requires_grad=True)
A = torch.rand(N, N, dtype=torch.float32)

def f(x):
    return torch.dot(x, (A @ x))

fwd_time = 1e20
hess_time = 1e20

for i in range(num_iter):
    start = time.time()
    y = f(x)
    fwd = time.time()
    hess = torch.autograd.functional.hessian(f, x)
    end = time.time()

    if fwd - start < fwd_time:
        fwd_time = fwd - start
    if end - fwd < hess_time:
        hess_time = end - fwd

print('Minimum forward time:', fwd_time)
print('Minimum Hessian time:', hess_time)
print('Ratio:', hess_time / fwd_time)