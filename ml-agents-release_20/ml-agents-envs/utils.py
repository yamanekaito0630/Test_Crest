import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np
import pickle

from mpl_toolkits.mplot3d import axes3d


def coordinates_heatmap_creator(path, coordinates, n_recode, range=4, resize=36, vmax=10):
    plt.rcParams.update({'font.size': 40})
    x, z, y = coordinates / 9.2

    plt.figure(figsize=(13, 13))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[-range, range], [-range, range]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=vmax)

    plt.colorbar(label='Frequency', shrink=0.81)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path + 'xy_heatmap_{}.png'.format(n_recode))

    plt.figure(figsize=(8, 8))
    heatmap, xedges, yedges = np.histogram2d(x, z, bins=50, range=[[-range, range], [-range, range]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=40)

    plt.colorbar(label='Frequency', shrink=0.81)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(path + 'xz_heatmap_{}.png'.format(n_recode))
    
    plt.clf()
    plt.close()


def at_scatter_creator(path, n_recode, apm, fontsize=23):
    r, g, b, imp = apm
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(r, g, b,
                         s=1,
                         c=imp,
                         cmap='viridis',
                         norm=mcolors.LogNorm())
    # scatter = ax.scatter(r, g, b,
    #                      s=1,
    #                      c=imp,
    #                      cmap='viridis')

    ax.set_xlabel('Red', rotation=None, labelpad=15, fontsize=fontsize)
    ax.set_ylabel('Green', rotation=None, labelpad=15, fontsize=fontsize)
    ax.set_zlabel('Blue', rotation=90, labelpad=15, fontsize=fontsize)

    ax.xaxis.set_ticks([30, 60, 90, 120, 150, 180, 210, 240])
    ax.yaxis.set_ticks([30, 60, 90, 120, 150, 180, 210, 240])
    ax.zaxis.set_ticks([30, 60, 90, 120, 150, 180, 210, 240])

    plt.tight_layout()
    cbar = plt.colorbar(scatter, shrink=0.7, pad=0.1)
    cbar.set_label('Importance', fontsize=fontsize)
    plt.savefig(path + 'at_scatter_{}.png'.format(n_recode))
    
    plt.clf()
    plt.close()


for i in [1, 2, 4, 8, 12]:
    for j in [1, 2, 4, 8, 12]:
        for k in range(3):
            c = pickle.load(open('log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/robots_coordinates_{}.pkl'.format(i, j, k), 'rb'))
            if j == 1:
                vmax = 8
            elif j == 2:
                vmax = 10
            elif j == 4:
                vmax = 12
            elif j == 8:
                vmax = 15
            elif j == 12:
                vmax = 20
            else:
                vmax = 10
            coordinates_heatmap_creator(path='log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/'.format(i, j), coordinates=c, n_recode=k, vmax=vmax)
            
