import model_helpers
import numpy as np

ma = np.zeros((3,3,3))

c=1
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            ma[i][j][k] = c
            c+=1

ma = np.random.randint(27, size=(4, 3,3))
print ma
print "#####"
x,y,z = model_helpers.get_min_3dmatrix(ma)

print (x,y,z)