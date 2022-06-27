import time
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    #plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()


def plot_original_face(dataset, idx, rotate=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data = dataset[idx]
    ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], c=data.pos[:, 2], cmap='viridis', linewidth=0.5)

    ax.view_init(100, -90)

    if rotate:
        for angle in range(0, 360):
            ax.view_init(0, angle, "y")
            plt.draw()
            plt.pause(.001)
    else:
        plt.show()

def plot_sequence(dataset, idx, finish):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(100, -90)

    total = finish - idx

    for i in range(0, total):
        x = ax.scatter(dataset[idx].pos[:, 0], dataset[idx].pos[:, 1], dataset[idx].pos[:, 2], c=dataset[idx].pos[:, 2], cmap='viridis', linewidth=0.5)
        idx+=1
        plt.draw()
        plt.pause(.001)
        x.remove()

def plot_minimized_face_(dataset, idx, rotate = False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    new_sample = torch.stack([p for p in dataset.get(idx).pos if p[2] > 0.01])
    ax.scatter(new_sample[:, 0], new_sample[:, 1], new_sample[:, 2], c=new_sample[:, 2], cmap='hot', linewidth=0.5)
    ax.view_init(100, -90)

    if rotate:
        for angle in range(0, 360):
            ax.view_init(0, angle, "y")
            plt.draw()
            plt.pause(.001)
    else:
        plt.show()

def convert_dataset_sizes(dataset, num_elements):
    labels = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}
    samples = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[]}

    for d in dataset:
        labels[int(d.y)] += 1
        samples[int(d.y)].append(d)

    random.seed(230)

    #Insert elements to classes with low number of samples
    new_dataset = []
    for lb in labels:
        if labels[lb] < num_elements:
            difference = num_elements - labels[lb]
            random_samples = random.choices(samples[lb], k = difference)
            new_dataset.extend(random_samples)
            new_dataset.extend(samples[lb])
        else:
            random_samples = random.sample(samples[lb], k=num_elements)
            new_dataset.extend(random_samples)
    return new_dataset