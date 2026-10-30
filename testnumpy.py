import numpy as np
import beams
import time

from beams import (
    ss_center_point,
    ss_center_pointloop,
    ss_off_center_point,
    ss_off_center_pointloop,
    ss_udl,
    ss_udlloop,
    cantilever_point,
    cantilever_pointloop,
    cantilever_udl,
    cantilever_udlloop
)


L = 2
P = 1000
w = 500
E = 200e9
I = 4.1666667e-6
a = 0.3 * L


x = np.linspace(0, L, 101)


numpy_results = [
    ss_center_point(x, L, P, E, I),
    ss_off_center_point(x, L, P, E, I, a),
    ss_udl(x, L, w, E, I),
    cantilever_point(x, L, P, E, I),
    cantilever_udl(x, L, w, E, I)
]


loop_results = [
    np.array([ss_center_pointloop(pos, L, P, E, I) for pos in x]),
    np.array([ss_off_center_pointloop(pos, L, P, E, I, a) for pos in x]),
    np.array([ss_udlloop(pos, L, w, E, I) for pos in x]),
    np.array([cantilever_pointloop(pos, L, P, E, I) for pos in x]),
    np.array([cantilever_udlloop(pos, L, w, E, I) for pos in x])
]


names = [
    "Simply supported centre point",
    "Simply supported off-centre point",
    "Simply supported UDL",
    "Cantilever point",
    "Cantilever UDL"
]

for i in range(len(names)):
    if np.allclose(numpy_results[i], loop_results[i]):
        print(names[i], ": PASSED")
    else:
        print(names[i], ": FAILED")

print("\nChecking theoretical maximum deflections:")


y = beams.ss_center_point(x, L, P, E, I)
expected = P * L**3 / (48 * E * I)
print("Centre point:", np.isclose(max(y), expected))

y = beams.ss_udl(x, L, w, E, I)
expected = 5 * w * L**4 / (384 * E * I)
print("UDL:", np.isclose(max(y), expected))


y = beams.cantilever_point(x, L, P, E, I)
expected = P * L**3 / (3 * E * I)
print("Cantilever point:", np.isclose(max(y), expected))

y = beams.cantilever_udl(x, L, w, E, I)
expected = w * L**4 / (8 * E * I)
print("Cantilever UDL:", np.isclose(max(y), expected))        

print("\nChecking boundary conditions:")

print("Centre point:", np.isclose(beams.ss_center_point(0, L, P, E, I), 0),
      np.isclose(beams.ss_center_point(L, L, P, E, I), 0))

print("Off-centre point:", np.isclose(beams.ss_off_center_point(0, L, P, E, I, a), 0),
      np.isclose(beams.ss_off_center_point(L, L, P, E, I, a), 0))

print("UDL:", np.isclose(beams.ss_udl(0, L, w, E, I), 0),
      np.isclose(beams.ss_udl(L, L, w, E, I), 0))

print("Cantilever point:", np.isclose(beams.cantilever_point(0, L, P, E, I), 0))

print("Cantilever UDL:", np.isclose(beams.cantilever_udl(0, L, w, E, I), 0))

print("\nChecking off-centre transition:")

left = beams.ss_off_center_pointloop(a, L, P, E, I, a)
right = beams.ss_off_center_pointloop(a, L, P, E, I, a)

print("Transition continuity:", np.isclose(left, right))

x_big = np.linspace(0, L, 1000000)

start = time.time()
[beams.ss_center_pointloop(pos, L, P, E, I) for pos in x_big]
loop_time = time.time() - start

start = time.time()
beams.ss_center_point(x_big, L, P, E, I)
numpy_time = time.time() - start

print("Loop time:", loop_time)
print("NumPy time:", numpy_time)
print("Speedup:", loop_time / numpy_time)