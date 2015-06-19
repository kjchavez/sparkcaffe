import numpy as np
import numexpr as ne
import time

param_count = [1000000, 5000000, 10000000]

print "Num Params".ljust(20), " copy time (ms) "
print "-"*80
for n in param_count:
    a = np.random.rand(n)
    b = np.empty_like(a, dtype=a.dtype)
    tic = time.time()
    #b[:] = a
    np.copyto(b, a, casting='no')
    toc = time.time()
    assert(np.sum(a - b) == 0)
    print str(n).ljust(20), 1000*(toc - tic), "ms"
