import matplotlib.pyplot as plt
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

x, z, y = pickle.load(open('log/at4/round_im_1_robo_slurm_at4/robots_coordinates_0.pkl', 'rb'))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("Helix", size=20)

ax.set_xlabel("x", size=14)
ax.set_ylabel("y", size=14)
ax.set_zlabel("z", size=14)

ax.plot(x, y, z, color='red')
plt.show()
