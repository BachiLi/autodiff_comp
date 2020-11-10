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

fwd_time = 1e20
jvp_time = 1e20

for i in range(num_iter):
    start = time.time()
    y = jf(A)
    fwd = time.time()
    jvp = jdf(A, Ad)
    end = time.time()

    if fwd - start < fwd_time:
        fwd_time = fwd - start
    if end - fwd < jvp_time:
        jvp_time = end - fwd

print('Minimum forward time:', fwd_time)
print('Minimum jvp time:', jvp_time)
print('Ratio:', jvp_time / fwd_time)