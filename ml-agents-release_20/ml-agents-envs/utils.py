import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import webcolors
import pickle

from statistics import variance, mean
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def coordinates_heatmap_creator(path, coordinates, n_recode, range=4, resize=36, vmax=10):
    plt.rcParams.update({'font.size': 30})
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
    plt.cla()


def at_scatter_creator(path, n_recode, apm, fontsize=37):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})

    r, g, b, _ = apm
    colors = np.vstack((r, g, b)).T / 255.0
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(r, g, b,
                         s=1,
                         c=colors)

    ax.set_xlabel('Red', rotation=None, labelpad=30)
    ax.set_ylabel('Green', rotation=None, labelpad=30)
    ax.set_zlabel('Blue', rotation=90, labelpad=30)

    ax.xaxis.set_ticks([0, 31, 63, 95, 127, 159, 191, 223, 255])
    ax.yaxis.set_ticks([0, 31, 63, 95, 127, 159, 191, 223, 255])
    ax.zaxis.set_ticks([0, 31, 63, 95, 127, 159, 191, 223, 255])
    ax.zaxis.set_tick_params(pad=10)

    plt.tight_layout()
    plt.savefig(path + 'at_scatter_{}.png'.format(n_recode))
    plt.clf()
    plt.close()
    plt.cla()


def coordinates_3d(path, n_recode, coordinates, n_robo, fontsize=60):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.xmargin'] = 0

    x, _, y = coordinates / 9.2
    _, z, _ = (coordinates + 10) / 5
    fig = plt.figure(figsize=(15, 13))
    ax = Axes3D(fig)
    fig.add_axes(ax)

    for i in range(n_robo):
        ax.plot(x[i::n_robo][1:], y[i::n_robo][1:], (z[i::n_robo] + 4.0)[1:], lw=5)

    ax.set_xlabel('X', rotation=None, labelpad=40, fontsize=fontsize)
    ax.set_ylabel('Y', rotation=None, labelpad=35, fontsize=fontsize)
    ax.set_zlabel('Z', rotation=None, labelpad=20, fontsize=fontsize)

    ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
    ax.yaxis.set_ticks([-4, -2, 0, 2, 4])
    ax.zaxis.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    plt.savefig(path + '3d_coordinates_{}.png'.format(n_recode))

    plt.clf()
    plt.close()
    plt.cla()


def z_t_creator(path, n_recode, coordinates, n_robo, timesteps=1000, fontsize=64, width=3):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.xmargin'] = 0

    t = np.linspace(1, timesteps, timesteps - 1)

    _, z, _ = (coordinates + 25) / 5
    print(z)
    fig = plt.figure(figsize=(18, 13))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(length=9, width=width)
    ax.spines["top"].set_linewidth(width)
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)

    for i in range(n_robo):
        plt.plot(t, z[i::n_robo][1:], lw=8)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Z', rotation=90)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    plt.tight_layout()
    plt.savefig(path + 't_z_{}.png'.format(n_recode))

    plt.clf()
    plt.close()
    plt.cla()


def save_movement_amount_of_z(path, coordinates, n_robo, timesteps=1000):
    _, z, _ = (coordinates + 25) / 5
    robots_z_amounts = []

    for i in range(n_robo):
        a_robot_z_amounts = []
        zs = z[i::n_robo][1:]
        for t in range(timesteps - 2):
            z_amount = abs(zs[t + 1] - zs[t])
            a_robot_z_amounts.append(z_amount)
        a_robot_episode_z_amount = sum(a_robot_z_amounts)
        robots_z_amounts.append(a_robot_episode_z_amount)

    with open(path + 'movement_amount_of_z.txt', mode='w') as f:
        res = np.mean(robots_z_amounts)
        f.write(str(res))

    return res
        

def save_variance_of_z(path, coordinates, n_robo, timesteps=1000):
    _, z, _ = (coordinates + 25) / 5
    z_variances = []

    zs = z[n_robo:]
    for t in range(timesteps - 1):
        ss = t * n_robo
        ee = ss + n_robo
        print(zs[ss:ee])
        v = np.var(zs[ss:ee])
        z_variances.append(v)

    with open(path + 'variance_of_z.txt', mode='w') as f:
        res = mean(z_variances)
        f.write(str(res))

    return res
            

def network_analyzer(path, nodes, links, n_robo, n_recode, timesteps=1000, fontsize= 20):
    for t in range(timesteps):
        for key, value in nodes[t].items():
            value = [e / 9.2 for e in value]
            value[1] += 4
            nodes[t][key] = value
            
    colors = ['red', 'blue', 'yellow', 'green', 'black', 'orange', 'purple', 'wheat', 'pink', 'grey', 'plum',
              'slateblue', 'darkgreen', 'darkred', 'olive', 'aqua', 'blueviolet', 'brown', 'sienna', 'tan']
    labels = []
    G = nx.DiGraph()
    for node in range(n_robo):
        G.add_node(node)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    
    def update(t, nodes, links, labels):
        plt.cla()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(0, 5)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.set_title("timestep={}".format(t))
        
        t = 2 if t < 2 else t
    
        edges = links[t]
        for i, e in enumerate(edges):
            if e[0] == e[1]:
                edges.pop(i)
        
        if len(edges) == 0:
            edges = links[t]
        
        G.add_edges_from(edges)
        pos_ary = np.array([nodes[t][n] for n in G])
        lst_m = greedy_modularity_communities(G)
        color_map_m =['black'] * nx.number_of_nodes(G)
        counter = 0
        for c in lst_m:
            for n in c:
                color_map_m[n] = colors[counter]
            counter = counter + 1
            
        color_counts = Counter(color_map_m)
        c_count_ary = [color_counts[color] for color in set(color_map_m)]
        labels += c_count_ary
        if t == timesteps - 1:
            with open(path + "cluster_info_{}.txt".format(n_recode), "w") as f:
                for num in range(n_robo):
                    f.write("{n}robo,{c}".format(n=num + 1, c=labels.count(num + 1)) + "\n")
            
            
        for i, pos in enumerate(pos_ary):
            ax.plot(pos[0], pos[2], pos[1], marker='o', color=color_map_m[i])
            
        for e in G.edges:
            node0_pos = nodes[t][e[0]]
            node1_pos = nodes[t][e[1]]
            x, dx = node0_pos[0], node1_pos[0] - node0_pos[0]
            y, dy = node0_pos[2], node1_pos[2] - node0_pos[2]
            z, dz = node0_pos[1], node1_pos[1] - node0_pos[1]
            ax.quiver(x, y, z, dx, dy, dz, color='black', arrow_length_ratio=0.3, linewidth=0.5)
        G.remove_edges_from(edges)
        
    ani = animation.FuncAnimation(fig, update, timesteps, fargs=(nodes, links, labels), interval=20)
    ani.save(path + "network_analysis_{}.mp4".format(n_recode), writer="ffmpeg")
    
    
def similarity_generator(actions, color_counts, timesteps=100):
    ress = []
    res = []
    sims = {}
    t_actions = list(zip(*actions))
    t_color_counts = list(zip(*color_counts))
    for i, a_s in enumerate(t_actions):
        sims[i] = {}
        for j, c_s in enumerate(t_color_counts):
            print(len(c_s))
            print(c_s)
            sims[i][j] = cos_sim(a_s, min_max_norm(c_s))

        top_key = max(sims[i], key=sims[i].get)
        res.append((i, int(top_key)))
        res.append(a_s)
        res.append(t_color_counts[int(top_key)])

        ress.append(res)
        res = []
    
    # res: [(action_index, top_key), [actions][colors]]
    return ress, sims

def save_sim_matrix(path, matrix, n_recode):
    row = []
    col = []
    for a_k, a_v in matrix.items():
        for r_k, r_v in a_v.items():
            row.append(r_v)

        col.append(row)
        row = []
    
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(col)
    fig.colorbar(im, ax=ax)
    plt.savefig(path + 'action_color_sim_matrix_{}.png'.format(n_recode))
    plt.cla()
    
        
def min_max_norm(l):
    min_value = min(l)
    max_value = max(l)
    if min_value == max_value:
        return l
    
    scaled_list = [2 * ((x - min_value) / (max_value - min_value)) - 1 for x in l]
    return scaled_list


    
def cos_sim(v1, v2):
    if np.dot(v1, v2) == 0.0 or (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0.0:
        return -1.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def save_action_color_series(path, action_color_sims, n_recode, timesteps=1000, fontsize=20):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.xmargin'] = 0

    t = np.linspace(1, timesteps, timesteps - 1)
    fig = plt.figure(figsize=(18, 13))
    xticks = list(range(0, timesteps + 1, 200))
    yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for sim in action_color_sims:
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        plot1 = ax1.plot(t, sim[1][1:], lw=3)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Action[{}]'.format(sim[0][0]))
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)

        plot2 = ax2.plot(t, sim[2][1:], lw=3)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Region[{}]'.format(sim[0][1]))
        ax2.set_xticks(xticks)

        plt.tight_layout()
        plt.savefig(path + 'action[{}]_region[{}]_series_{}.png'.format(sim[0][0], sim[0][1], n_recode))
        plt.clf()
    plt.cla()


def save_action_color_fourier(path, action_color_sims, n_recode, timesteps=1000, dt=0.02, fontsize=20):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.xmargin'] = 0

    N = timesteps - 1
    freq = np.fft.fftfreq(N, d=dt)
    fig = plt.figure(figsize=(18, 13))
    for sim in action_color_sims:
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        a_fft = np.fft.fft(sim[1][1:])
        a_amp = abs(a_fft / (N / 2))
        ax1.plot(freq[1:int(N / 2)], a_amp[1:int(N / 2)], lw=3)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Amplitude of action[{}]'.format(sim[0][0]))

        r_fft = np.fft.fft(sim[2][1:])
        r_amp = abs(r_fft / (N / 2))
        ax2.plot(freq[1:int(N / 2)], r_amp[1:int(N / 2)], lw=3)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Amplitude of region[{}]'.format(sim[0][1]))

        plt.tight_layout()
        plt.savefig(path + 'action[{}]_region[{}]_fourier_{}.png'.format(sim[0][0], sim[0][1], n_recode))
        plt.clf()
    plt.cla()
    
    
def rgb_analyzer(path, apm):
    colors = np.array([
        [84, 84, 84], # black
        [180, 180, 180], # white
        [0, 222, 255], # aqua
        [255, 255, 0], # yellow
        [255, 0, 210], # pink
        [255, 0, 0], # red
        [0, 0, 255], # blue
        [255, 148, 8] # orange
    ])

    apm = np.array(apm)
    rgb_list = apm.T
    colors = colors / 255
    percents = rgb_classifier(rgb_list)
    print(percents)
    tup = zip(colors, percents)
    sorted_tup = sorted(tup, key=lambda n: n[1], reverse=True)
    sorted_colors = [c for c,p in sorted_tup]
    sorted_percents = [p for c,p in sorted_tup]
    
    label_names = ["{}%".format(v) for v in sorted_percents]
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.pie(sorted_percents, colors=sorted_colors, counterclock=False, startangle=90)
    plt.legend(label_names, bbox_to_anchor=(0.9, 0.7))
    plt.savefig(path + 'rgb_percents_.png')
    plt.cla()
    
    
def test_rgb_analyzer(path, apm):
    init_centers = np.array([
        [80, 80, 80], # black
        [200, 200, 200], # white
        [0, 222, 255], # aqua
        [255, 255, 0], # yellow
        [255, 0, 210], # pink
        [255, 0, 0], # red
        [0, 0, 255], # blue
        [255, 148, 8] # orange
    ])

    r, g, b, _ = apm
    lst = [r,g,b]
    apm = np.array(lst)
    rgb_list = apm.T
    
    labels = pairwise_distances_argmin(rgb_list, init_centers)
    percents = cluster_percents(labels)
    
    label_names = ["{}%".format(v) for v in percents]
    plt.pie(percents, colors=init_centers / 255, counterclock=False, startangle=90)
    plt.legend(label_names, bbox_to_anchor=(0.9, 0.7))
    plt.savefig(path + 'rgb_percents_.png')
    plt.cla()

    
def cluster_percents(labels):
    total = len(labels)
    percents = []
    for i in set(labels):
        percent = (np.count_nonzero(labels == i) / total) * 100
        percents.append(round(percent, 2))
    return percents


def get_top_color(rs, gs, bs):
    colors = np.array([
        [84, 84, 84], # black
        [180, 180, 180], # white
        [0, 222, 255], # aqua
        [255, 255, 0], # yellow
        [255, 0, 210], # pink
        [255, 0, 0], # red
        [0, 0, 255], # blue
        [255, 148, 8] # orange
    ])

    rgb_list = np.array([rs, gs, bs])
    rgb_list = rgb_list.T
    percents = rgb_classifier(rgb_list)
    tup = zip(colors, percents)
    sorted_tup = sorted(tup, key=lambda n: n[1], reverse=True)
    sorted_colors = [c for c, p in sorted_tup]
    top_color = sorted_colors[0]

    return top_color


def get_scatter_of_actions(path, obs_actions, fontsize=20):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(8, 8))

    for point in obs_actions:
        plt.scatter(point[0], point[1], c=point[2] / 255, s=3)

    plt.xlabel('action for turning')
    plt.ylabel('action for z move')
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.savefig(path + 'obs_actions.png')
    plt.cla()


def get_scatter_of_leds(path, obs_leds, fontsize=20):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(8, 8))

    for point in obs_leds:
        plt.scatter(point[0], point[1], c=point[2] / 255, s=3)

    plt.xlabel('front LED')
    plt.ylabel('lear LED')
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.savefig(path + 'obs_leds.png')
    plt.cla()


def rgb_classifier(rgb_list):
    n_black = 0
    n_white = 0
    n_aqua = 0
    n_yellow = 0
    n_pink = 0
    n_red = 0
    n_blue = 0
    n_orange = 0

    blacks = ["gainsboro","lightgrey","silver","darkgray","gray","dimgray","lightslategray","slategray","darkslategray","black"]
    whites = ["lavender","lightcyan","white","snow","honeydew","mintcream","azure","aliceblue","ghostwhite","whitesmoke"]
    aquas = ["lightseagreen","darkcyan","teal","aqua","cyan","paleturquoise","aquamarine","turquoise"]
    yellows = ["gold","yellow","lightyellow","lemonchiffon","lightgoldenrodyellow","palegoldenrod","khaki","darkkhaki","goldenrod"]
    pinks = ["pink", "lightpink","hotpink","deeppink","mediumvioletred","palevioletred","thistle","plum","violet","orchid","fuchsia","magenta","mediumorchid","",""]
    reds = ["indianred", "lightcoral","salmon","darksalmon","lightsalmon","crimson","red","firebrick","darkred","brown"]
    blues = ["dodgerblue","cornflowerblue","royalblue","blue","mediumblue","darkblue","navy","midnightblue"]
    oranges = ["coral","tomato","orangered","darkorange","orange","sandybrown"]

    for rgb in rgb_list:
        e = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        color_name = get_color_name(e)
        if color_name in blacks:
            n_black += 1
        elif color_name in whites:
            n_white += 1
        elif color_name in aquas:
            n_aqua += 1
        elif color_name in yellows:
            n_yellow += 1
        elif color_name in pinks:
            n_pink += 1
        elif color_name in reds:
            n_red += 1
        elif color_name in blues:
            n_blue += 1
        elif color_name in oranges:
            n_orange += 1

    labels = [n_black, n_white, n_aqua, n_yellow, n_pink, n_red, n_blue, n_orange]
    percents = [round(label * 100 / np.sum(labels), 1) for label in labels]

    return percents


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_tuple):
    try:
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        return closest_color(rgb_tuple)

