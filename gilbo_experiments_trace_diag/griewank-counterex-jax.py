

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
#import tensorflow as tf
import time
import math


randkey = jax.random.PRNGKey(0)



def f(x):
  A = jnp.diag(x)
  v = jnp.dot(A[:,0],A[0,:])
  return v

f  = jit(f)
Df = jit(grad(f))

N = 3
x = jax.random.uniform(randkey,[N])
f(x)
Df(x)

def run_exp_0(N):
  x = jax.random.uniform(randkey,[N])

  f(x).block_until_ready()
  Df(x).block_until_ready()

  start_time  = time.perf_counter()
  v = f(x).block_until_ready()
  stop_time   = time.perf_counter()
  t = stop_time - start_time

  start_time  = time.perf_counter()
  g = Df(x).block_until_ready()
  stop_time   = time.perf_counter()
  dt = stop_time - start_time

  return (dt / t), t, dt

for K in (1,2,4,8,10,16):
  N = K*2000
  ratio, t, dt = run_exp_0(N)
  print(f"Griewank ratio for N = {N}:  {ratio}")
  print(f" : {t} {dt}")

print("\n----\n")


"""
def run_exp_0(N,K):
  @tf.function(
    experimental_compile=True,
    input_signature=[tf.TensorSpec(shape=[N], dtype=tf.float32)]
  )
  def f(x):
    grads = None
    with tf.GradientTape() as tape:
      tape.watch(x)
      A     = tf.linalg.diag(x)
      v     = (tf.tensordot(A[:,0], A[0,:], 1))
      #v     = tf.math.reduce_sum( tf.linalg.diag_part(A, k=0) )
      for i in range(1,K):
        #A   = tf.linalg.set_diag(A,y[i])
        v   += (tf.tensordot(A[:,i], A[i,:], 1))
        #v   = v + tf.math.reduce_sum( tf.linalg.diag_part(A, k=i) )
        #v   = v + tf.linalg.trace(A)
    return v

  @tf.function(
    experimental_compile=True,
    input_signature=[tf.TensorSpec(shape=[N], dtype=tf.float32)]
  )
  def Df(x):
    grads = None
    with tf.GradientTape() as tape:
      tape.watch(x)
      A     = tf.linalg.diag(x)
      v     = (tf.tensordot(A[:,0], A[0,:], 1))
      #v     = tf.math.reduce_sum( tf.linalg.diag_part(A, k=0) )
      for i in range(1,K):
        #A   = tf.linalg.set_diag(A,y[i])
        v   += (tf.tensordot(A[:,i], A[i,:], 1))
        #v   = v + tf.math.reduce_sum( tf.linalg.diag_part(A, k=i) )
        #v   = v + tf.linalg.trace(A)
      grads = tape.gradient(v, x)
    return grads

  x = tf.random.uniform([N])

  f(x)
  Df(x)

  start_time  = time.perf_counter()
  v = f(x)
  stop_time   = time.perf_counter()
  t = stop_time - start_time

  start_time  = time.perf_counter()
  g = Df(x)
  stop_time   = time.perf_counter()
  dt = stop_time - start_time

  return (dt / t), t, dt
"""






