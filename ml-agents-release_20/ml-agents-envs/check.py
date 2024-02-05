import matplotlib.pyplot as plt
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)

x=[1,2,3,4,5]

for i in range(3):
    y=np.full(5,i)
    z=np.random.rand(5)

    ax.plot(x,y,z)

plt.show()
