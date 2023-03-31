import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from modules.game import GameModule

import configs

from kmeans_pytorch import kmeans, kmeans_predict



def recordGameData(num_agent, num_prey):
    agent = torch.load('models/29-03-2023 1622 easy1211.pt')
    agent.reset()
    agent.train(False)

    con = configs.default_game_config
    game = GameModule(con, num_agent, num_prey)

    _, timesteps = agent(game)

    data = torch.Tensor(configs.DEFAULT_TIME_HORIZON, con.batch_size, num_agent+num_prey, 2)
    for val in range(configs.DEFAULT_TIME_HORIZON):
        data[val] = timesteps[val]['locations']
    torch.save(data, f"data\{num_agent}{num_prey}.pt")

def kclusterGame(path, num):
    file = torch.load(f'data/{path}')
    gamedata = torch.flatten(file,end_dim=2)
    num_clusters = num
    device = torch.device('cpu')
    result = kmeans(
    X=gamedata, num_clusters=num_clusters, distance='euclidean', device=device
    )
    torch.save(result, f'data/cluster{path}')

def visualizeClusters(path):
    f1 = torch.load(f'data/{path}')
    f2 = torch.load(f'data/cluster{path}')
    cluster_ids_x_und, cluster_centers_und = f2
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers = cluster_centers_und.detach()
    x = torch.flatten(f1.detach(),end_dim=2)

    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    plt.axis([-4, 20, -4, 20])
    plt.tight_layout()
    plt.show()

kclusterGame('21.pt', 20)
visualizeClusters('21.pt')