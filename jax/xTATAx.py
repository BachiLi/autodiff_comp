# Compute the Hessian of x^TAx w.r.t. x
import jax
import jax.numpy as np
import time

num_iter = 10
N = 1000

key = jax.random.PRNGKey(1234)
x = jax.random.uniform(key, shape=[N], dtype=np.float32)
A = jax.random.uniform(key, shape=[N, N], dtype=np.float32)
Ad = jax.random.uniform(key, shape=[N, N], dtype=np.float32)

def f(A):
    return np.dot(x, (np.transpose(A) @ (A @ x)))

def df(A, Ad):
    return jax.jvp(f, [A], [Ad])

jf = jax.jit(f)
jdf = jax.jit(df)

min_fwd_time = 1e20
min_jvp_time = 1e20

avg_fwd_time = 0
avg_jvp_time = 0

for i in range(num_iter + 1):
    start = time.time()
    y = jf(A).block_until_ready()
    fwd = time.time()
    jvp = jdf(A, Ad)
    jvp[0].block_until_ready()
    jvp[1].block_until_ready()
    end = time.time()

    if i > 0:
        avg_fwd_time += fwd - start
        avg_jvp_time += end - fwd
        if fwd - start < min_fwd_time:
            min_fwd_time = fwd - start
        if end - fwd < min_jvp_time:
            min_jvp_time = end - fwd

print('Average forward time:', avg_fwd_time / num_iter)
print('Average jvp time:', avg_jvp_time / num_iter)
print('Ratio of average:', avg_jvp_time / avg_fwd_time)
print('Minimum forward time:', min_fwd_time)
print('Minimum jvp time:', min_jvp_time)
print('Ratio of minimum:', min_jvp_time / min_fwd_time)