import numpy as np
import tensorflow as tf
import time
import math

tf.compat.v1.enable_eager_execution()

def run_exp_0(N):
  @tf.function(
    experimental_compile=True,
    input_signature=[tf.TensorSpec(shape=[N,N], dtype=tf.float32)]
  )
  def f(A):
    grads = None
    with tf.GradientTape() as tape:
      tape.watch(A)
      AAt   = tf.matmul( A, tf.transpose(A) )
      trAA  = tf.linalg.trace(AAt)
      grads = tape.gradient(trAA, A)
    return trAA, grads/2.0

  A = tf.random.uniform([N,N])

  f(A)

  start_time  = time.perf_counter()
  f(A)
  stop_time   = time.perf_counter()

  return stop_time - start_time



for N in (100,200,400,800,1200,1600):
  timing = run_exp_0(N)
  print(f"time            for N = {N}:  {timing}")
  print(f"sqrt(time)      for N = {N}:      {math.sqrt(timing)}")
  print(f"cube_root(time) for N = {N}:          {math.pow(timing,1.0/3.0)}")

