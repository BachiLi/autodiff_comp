

import numpy as np
import tensorflow as tf
import time
import math

tf.compat.v1.enable_eager_execution()

def run_diag_trace(N,K):
  def f(x):
    A     = tf.linalg.diag(x)
    v     = tf.linalg.trace(A)
    for i in range(1,K):
      v   += tf.linalg.trace(A)
    return v

  def Df(x):
    grads = None
    with tf.GradientTape() as tape:
      tape.watch(x)
      v     = f(x)
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


def xla_diag_trace(N,K):
  @tf.function(
    experimental_compile=True,
    input_signature=[tf.TensorSpec(shape=[N], dtype=tf.float32)]
  )
  def f(x):
    A     = tf.linalg.diag(x)
    v     = tf.linalg.trace(A)
    for i in range(1,K):
      v   += tf.linalg.trace(A)
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
      v     = tf.linalg.trace(A)
      for i in range(1,K):
        v   += tf.linalg.trace(A)
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

def xla_diag_dot(N,K):
  @tf.function(
    experimental_compile=True,
    input_signature=[tf.TensorSpec(shape=[N], dtype=tf.float32)]
  )
  def f(x):
    A     = tf.linalg.diag(x)
    v     = tf.tensordot(A[:,0], A[0,:], 1)
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
      v     = tf.tensordot(A[:,0], A[0,:], 1)
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

for K in range(1,11):
  N = 4000 * K
  ratio, t, dt = xla_diag_dot(N,K)
  print(f"Griewank ratio for N = {N}:  {ratio}, {t}, {dt}")
  #print(f" : {t} {dt}")

print("\n----\n")





