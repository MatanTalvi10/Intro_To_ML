import numpy as np
import skeleton

x = np.random.uniform(0,1,10)
x.sort()
print(type(x))

ass = skeleton.Assignment2()
res = skeleton.Assignment2.sample_from_D(ass,10)
print(res)
for_A, for_B = 0,0
print(for_B)