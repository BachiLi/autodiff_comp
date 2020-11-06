import torch
import time
import math

# Testing our logic with a small N
N = 3

A = torch.randn((N,N), requires_grad = True)

At = torch.transpose(A, 0, 1)

T = torch.trace(A @ At)

T.backward()

print(T)
print(A)
print(A.grad / 2.0)
# checks out that A matches its gradient



def time_test(N):
  start_time  = time.perf_counter()

  A = torch.randn((N,N), requires_grad = True)

  At = torch.transpose(A, 0, 1)

  T = torch.trace(A @ At)

  T.backward()

  stop_time  = time.perf_counter()

  return stop_time - start_time

for N in (100,200,400,800,1200,1600):
  timing = time_test(N)
  print(f"time            for N = {N}:  {timing}")
  print(f"sqrt(time)      for N = {N}:      {math.sqrt(timing)}")
  print(f"cube_root(time) for N = {N}:          {math.pow(timing,1.0/3.0)}")



print("\n----\n")

def time_test2(N):
  start_time  = time.perf_counter()

  A = torch.randn((N,N), requires_grad = True)

  T = torch.trace(A @ A)

  T.backward()

  stop_time  = time.perf_counter()

  return stop_time - start_time

for N in (100,200,400,800,1200,1600):
  timing = time_test2(N)
  print(f"time            for N = {N}:  {timing}")
  print(f"sqrt(time)      for N = {N}:      {math.sqrt(timing)}")
  print(f"cube_root(time) for N = {N}:          {math.pow(timing,1.0/3.0)}")



print("\n----\n")


def time_test3(N):
  start_time  = time.perf_counter()

  A = torch.randn((N,N), requires_grad = True)

  At = torch.transpose(A, 0, 1)

  T = torch.norm(A @ At)

  T.backward()

  stop_time  = time.perf_counter()

  return stop_time - start_time

for N in (100,200,400,800,1200,1600):
  timing = time_test2(N)
  print(f"time            for N = {N}:  {timing}")
  print(f"sqrt(time)      for N = {N}:      {math.sqrt(timing)}")
  print(f"cube_root(time) for N = {N}:          {math.pow(timing,1.0/3.0)}")





