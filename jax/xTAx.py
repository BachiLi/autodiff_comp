# Compute the Hessian of x^TAx w.r.t. x
import jax
import jax.numpy as np
import time

num_iter = 10
N = 6000

key = jax.random.PRNGKey(1234)
x = jax.random.uniform(key, shape=[N], dtype=np.float32)
A = jax.random.uniform(key, shape=[N, N], dtype=np.float32)

def f(x):
    return np.dot(x, (A @ x))

df = jax.hessian(f)

fwd_time = 1e20
hess_time = 1e20

for i in range(num_iter):
    start = time.time()
    y = f(x)
    fwd = time.time()
    hess = df(x)
    end = time.time()

    if fwd - start < fwd_time:
        fwd_time = fwd - start
    if end - fwd < hess_time:
        hess_time = end - fwd

print('Minimum forward time:', fwd_time)
print('Minimum Hessian time:', hess_time)
print('Ratio:', hess_time / fwd_time)