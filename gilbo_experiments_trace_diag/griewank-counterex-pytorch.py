import torch
import time
import math

# Testing our logic with a small N and k
N = 3
K = 3

x = torch.randn((N,), requires_grad = True)
A = torch.diag(x)

out = torch.trace(A)
for i in range(1,K):
  val = torch.trace(A)
  print(val)
  out += val

out.backward()

print(out)
print(x)
print(x.grad)
# x.grad should be the vector [K, K, K, ...]



def time_test(N,K):
  start_time  = time.perf_counter()

  x = torch.randn((N,), requires_grad = True)

  A = torch.diag(x)

  out = torch.trace(A)
  for i in range(1,K):
    out += torch.trace(A)

  mid_time   = time.perf_counter()

  out.backward()

  stop_time  = time.perf_counter()

  Dtime = stop_time - start_time
  Btime  = mid_time - start_time
  return Btime, Dtime, Dtime / Btime,


for K in (1,2,3,4,5,6,7,8,9,10):
  t, dt, ratio = time_test(20000,K)
  print(f"Griewank ratio for K = {K}:  {ratio} , {t} {dt}")

print("\n----\n")





