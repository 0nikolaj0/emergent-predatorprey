import torch
import torch.nn as nn

from modules.game import GameModule

import configs

from kmeans_pytorch import kmeans, kmeans_predict

agent = torch.load('models/29-03-2023 1622 easy1211.pt')
agent.reset()
agent.train(False)

con = configs.default_game_config

def recordGameData(num_agent, num_prey):
    game = GameModule(con, num_agent, num_prey)

    _, timesteps = agent(game)

    data = torch.FloatTensor(con.batch_size * configs.DEFAULT_TIME_HORIZON, num_agent+num_prey, 2)
    for keys in timesteps:
        torch.cat((data,keys['locations']),0)

    torch.save(data, f"data\{num_agent}{num_prey}.pt")

def kclusterGame(path):
    file = torch.load(path)
    gamedata = torch.flatten(file,end_dim=1)
    num_clusters = 8
    device = torch.device('cpu')
    cluster_ids_x, cluster_centers = kmeans(
    X=gamedata, num_clusters=num_clusters, distance='euclidean', device=device
    )
    return cluster_ids_x, cluster_centers

print(kclusterGame('data/21.pt'))
