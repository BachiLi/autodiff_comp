# Compute the Jacobian vector product (forward-mode) of x^TA^TAx w.r.t. A

import torch
import time

num_iter = 10
N = 100

x = torch.rand(N, dtype=torch.float32)
A = torch.rand(N, N, dtype=torch.float32, requires_grad=True)
Ad = torch.rand(N, N, dtype=torch.float32)

def f(A):
    return torch.dot(x, A.T @ (A @ x))

fwd_time = 1e20
jvp_time = 1e20

for i in range(num_iter):
    start = time.time()
    y = f(A)
    fwd = time.time()
    jvp = torch.autograd.functional.jvp(f, A, v=Ad)
    end = time.time()

    if fwd - start < fwd_time:
        fwd_time = fwd - start
    if end - fwd < jvp_time:
        jvp_time = end - fwd

print('Minimum forward time:', fwd_time)
print('Minimum jvp time:', jvp_time)
print('Ratio:', jvp_time / fwd_time)