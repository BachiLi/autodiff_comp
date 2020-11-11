# Compute the Hessian of x^TAx w.r.t. x
import jax
import jax.numpy as np
import time

num_iter = 10
N = 100

key = jax.random.PRNGKey(1234)
x = jax.random.uniform(key, shape=[N], dtype=np.float32)
A = jax.random.uniform(key, shape=[N, N], dtype=np.float32)

def f(x):
    return np.dot(x, (A @ x))

df = jax.hessian(f)

jf = jax.jit(f)
jdf = jax.jit(df)

min_fwd_time = 1e20
min_hess_time = 1e20

avg_fwd_time = 0
avg_hess_time = 0

for i in range(num_iter + 1):
    start = time.time()
    y = jf(x).block_until_ready()
    fwd = time.time()
    hess = jdf(x).block_until_ready()
    end = time.time()

    if i > 0:
        avg_fwd_time += fwd - start
        avg_hess_time += end - fwd
        if fwd - start < min_fwd_time:
            min_fwd_time = fwd - start
        if end - fwd < min_hess_time:
            min_hess_time = end - fwd

print('Average forward time:', avg_fwd_time / num_iter)
print('Average Hessian time:', avg_hess_time / num_iter)
print('Ratio of average:', avg_hess_time / avg_fwd_time)
print('Minimum forward time:', min_fwd_time)
print('Minimum Hessian time:', min_hess_time)
print('Ratio of minimum:', min_hess_time / min_fwd_time)
