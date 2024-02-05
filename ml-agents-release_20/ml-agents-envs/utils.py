import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D


def coordinates_heatmap_creator(path, coordinates, n_recode, range=4, resize=36, vmax=10):
    plt.rcParams.update({'font.size': 40})
    plt.rcParams.update({'font.family': 'Times New Roman'})

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


def at_scatter_creator(path, n_recode, apm, fontsize=40):
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

    ax.set_xlabel('Red', rotation=None, labelpad=30, fontsize=fontsize)
    ax.set_ylabel('Green', rotation=None, labelpad=30, fontsize=fontsize)
    ax.set_zlabel('Blue', rotation=90, labelpad=30, fontsize=fontsize)

    ax.xaxis.set_ticks([30, 60, 90, 120, 150, 180, 210, 240])
    ax.yaxis.set_ticks([30, 60, 90, 120, 150, 180, 210, 240])
    ax.zaxis.set_ticks([30, 60, 90, 120, 150, 180, 210, 240])

    plt.tight_layout()
    cbar = plt.colorbar(scatter, shrink=0.7, pad=0.1)
    cbar.set_label('Importance', fontsize=fontsize)
    plt.savefig(path + 'at_scatter_{}.png'.format(n_recode))
    
    plt.clf()
    plt.close()
    
    
def coordinates_3d(path, n_recode, coordinates, n_robo, fontsize=70):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.xmargin'] = 0
    
    x, z, y = coordinates / 9.2
    fig = plt.figure(figsize=(15, 13))
    ax = Axes3D(fig)
    fig.add_axes(ax)
    
    for i in range(n_robo):
        ax.plot((x[i::n_robo])[1:], (y[i::n_robo])[1:], ((z[i::n_robo])+abs(min(z))+1.0)[1:], lw=5)
                         


    ax.set_xlabel('X', rotation=None, labelpad=40, fontsize=fontsize)
    ax.set_ylabel('Y', rotation=None, labelpad=35, fontsize=fontsize)
    ax.set_zlabel('Z', rotation=None, labelpad=20, fontsize=fontsize)

    ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
    ax.yaxis.set_ticks([-4, -2, 0, 2, 4])
    ax.zaxis.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    plt.savefig(path + '3d_coordinates_{}.png'.format(n_recode))
    
    plt.clf()
    plt.close()


def z_t_creator(path, n_recode, coordinates, n_robo, fontsize=64, width=3):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})

    t = np.linspace(1, 1000, 999)

    _, z, _ = coordinates / 9.2
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(length=9, width=width)
    ax.spines["top"].set_linewidth(width)
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)

    for i in range(n_robo):
        plt.plot(t, ((z[i::n_robo])+abs(min(z))+1.0)[1:], lw=8)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Z', rotation=90)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([0.0, 1.0, 2.0, 3.0])

    plt.tight_layout()
    plt.savefig(path + 't_z_{}.png'.format(n_recode))
    
    plt.clf()
    plt.close()
    
    
    
for i in [1, 2, 4, 8, 12]:
    for j in [1, 2, 4, 8, 12]:
        for k in range(3):
            c = pickle.load(open('log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/robots_coordinates_{}.pkl'.format(i, j, k), 'rb'))
            coordinates_3d(path='log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/'.format(i, j), n_recode=k, coordinates=c, n_robo=j)
            z_t_creator(path='log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/'.format(i, j), n_recode=k, coordinates=c, n_robo=j)


#for i in [1, 2, 4, 8, 12]:
#    for j in [1, 2, 4, 8, 12]:
#        for k in range(3):
#            c = pickle.load(open('log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/robots_coordinates_{}.pkl'.format(i, j, k), 'rb'))
#            if j == 1:
#                vmax = 8
#            elif j == 2:
#                vmax = 10
#            elif j == 4:
#                vmax = 12
#            elif j == 8:
#                vmax = 15
#            elif j == 12:
#                vmax = 20
#            else:
#                vmax = 10
#            coordinates_heatmap_creator(path='log/at4/round_im_{}_robo_slurm_at4/eval_{}robo/'.format(i, j), coordinates=c, n_recode=k, vmax=vmax)
            
