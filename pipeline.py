import code
import torch
import configs
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from modules.game              import GameModule
from torchmetrics.functional   import kl_divergence
from kmeans_pytorch            import kmeans, kmeans_predict

def record_game_utter(num_agent, num_prey): #records game locations AND utterances for a single forward call
    agent = torch.load('models/14-04-2023 1417 easy1211.pt')
    agent.reset()
    agent.train(False)

    con = configs.default_game_config
    game = GameModule(con, num_agent, num_prey)

    _, timesteps = agent(game)

    locationdata = torch.Tensor(configs.DEFAULT_TIME_HORIZON, con.batch_size, num_agent+num_prey, 2)
    utterdata = torch.Tensor(configs.DEFAULT_TIME_HORIZON, con.batch_size, num_agent, con.vocab_size)
    for val in range(configs.DEFAULT_TIME_HORIZON):
        locationdata[val] = timesteps[val]['locations']
        utterdata[val] = timesteps[val]['utterances']
    return locationdata, utterdata

def get_game_metrics(locd, num_agent): #for each step in a game in a batch, computes the metrics #we define 2 metrics for a game:            #-mean of distance between agents                          #-mean of distance to the closest prey for each agent
    gamedata = torch.flatten(locd,end_dim=1)
    metric = torch.Tensor(gamedata.size()[0],2)
    for i in range(gamedata.size()[0]):
        agents = gamedata[i,:num_agent,:]
        metric[i,0] = torch.mean(torch.min(torch.cdist(agents,gamedata[i,num_agent:,:],1),1)[0])
        metric[i,1] = torch.mean(torch.cdist(agents,agents,1))
    return metric

def kcluster(file, num_clusters): #clusters game metrics
    device = torch.device('cpu')
    result = kmeans(
        X=file, num_clusters=num_clusters, distance='euclidean', device=device
    )
    return result

def visualize_clusters(f1, f2): #visualizes clustered game metrics
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
    locd, utterd = record_game_utter(num_agent, num_prey)
    metrics = get_game_metrics(locd, num_agent)
    clusters = kcluster(metrics, num_cluster)
    visualize_clusters(metrics, clusters)
    #return locd, utterd, metrics, clusters

pipeline(3,2,8)