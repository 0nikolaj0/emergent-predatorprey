import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from modules.game import GameModule

import configs

from kmeans_pytorch import kmeans, kmeans_predict

def recordGameData(num_agent, num_prey):
    agent = torch.load('models/29-03-2023 1622  easy1211.pt')
    agent.reset()
    agent.train(False)

    con = configs.default_game_config
    game = GameModule(con, num_agent, num_prey)

    _, timesteps = agent(game)

    data = torch.Tensor(configs.DEFAULT_TIME_HORIZON, con.batch_size, num_agent+num_prey, 2)
    for val in range(configs.DEFAULT_TIME_HORIZON):
        data[val] = timesteps[val]['locations']
    #print(data.size())
    torch.save(data, f"data/{num_agent}{num_prey}.pt")

def getGameMetrics(path):
    num_agent = int(path[0])
    num_prey  = int(path[1])
    gamedata = torch.flatten(torch.load(f'data/{path}'),end_dim=1)
    metric = torch.Tensor(gamedata.size()[0],2)
    for i in range(gamedata.size()[0]):
        agents = gamedata[i,:num_agent,:]
        metric[i,0] = torch.mean(torch.min(torch.cdist(agents,gamedata[i,num_agent:,:],1),1)[0])
        metric[i,1] = torch.mean(torch.cdist(agents,agents,1))
    torch.save(metric, f"data/metric{path}")

def kcluster(path, num_clusters):
    file = torch.load(f'data/{path}')
    device = torch.device('cpu')
    result = kmeans(
        X=file, num_clusters=num_clusters, distance='euclidean', device=device
    )
    torch.save(result, f'data/cluster{path}')

def visualizeClusters(path):
    f1 = torch.load(f'data/{path}')
    f2 = torch.load(f'data/cluster{path}')
    cluster_ids_x_und, cluster_centers_und = f2
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers = cluster_centers_und.detach()
    x = f1.detach()

    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='plasma')
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    plt.axis([torch.min(x[:,0])-1, torch.max(x[:,0])+1, torch.min(x[:,1])-1, torch.max(x[:,1])+1])
    plt.xlabel("mean distance agent to prey")
    plt.ylabel("mean distance between agents")
    plt.tight_layout()
    plt.show()


def pipeline(num_agent, num_prey, num_cluster):
    kcluster(f'metric{num_agent}{num_prey}.pt',num_cluster)
    visualizeClusters(f'metric{num_agent}{num_prey}.pt')

def visualizeMultiple(paths):
    fig, axs = plt.subplots(2,2)
    for i, path in enumerate(paths):
        ax = axs.flat[i]
        f1 = torch.load(f'data/{path}')
        f2 = torch.load(f'data/cluster{path}')
        cluster_ids_x_und, cluster_centers_und = f2
        cluster_ids_x = cluster_ids_x_und.detach()
        cluster_centers = cluster_centers_und.detach()
        x = f1.detach()

        ax.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='plasma')
        ax.scatter(
            cluster_centers[:, 0], cluster_centers[:, 1],
            c='white',
            alpha=0.6,
            edgecolors='black',
            linewidths=2
        )
        ax.axis([torch.min(x[:,0])-1, torch.max(x[:,0])+1, torch.min(x[:,1])-1, torch.max(x[:,1])+1])
        ax.set_title(path)
        ax.set_xlabel("μ dist agents to prey")
        ax.set_ylabel("μ dist between agents")
    fig.tight_layout()
    plt.show()

def plotLosses(path):
    fig, ax = plt.subplots()
    data = np.load(f'lossdata/{path}.npy')
    min_agents = int(path[0])
    max_agents = int(path[1])
    min_preys = int(path[2])
    max_preys = int(path[3])
    num_epochs = int(path[4:])

    for a in range(min_agents, max_agents + 1):
        for l in range(min_preys, max_preys + 1):
            plt.plot(np.arange(num_epochs), data[a-1][l-1], label=f"num_agents: {a} num_prey: {l}")
    ticks = np.arange(0,num_epochs,1)
    ax.set_xticks(ticks)
    plt.show()

plotLosses('12114')
#visualizeMultiple(['metric21.pt','metric22.pt', 'metric31.pt', 'metric32.pt'])
#x = torch.load('data/clustermetric21.pt')
#print(x)