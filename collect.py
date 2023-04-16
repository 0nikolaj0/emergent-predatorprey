import torch
import configs
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from modules.game              import GameModule
from torchmetrics.functional   import kl_divergence
from kmeans_pytorch            import kmeans, kmeans_predict

def record_game(num_agent, num_prey, save=False): #records game locations for a single forward call
    agent = torch.load('models/07-04-2023 1558 medium2312.pt')
    agent.reset()
    agent.train(False)

    con = configs.default_game_config
    game = GameModule(con, num_agent, num_prey)

    _, timesteps = agent(game)

    locationdata = torch.Tensor(configs.DEFAULT_TIME_HORIZON, con.batch_size, num_agent+num_prey, 2)
    for val in range(configs.DEFAULT_TIME_HORIZON):
        locationdata[val] = timesteps[val]['locations']
    if save:
        torch.save(locationdata, f"data/{num_agent}{num_prey}.pt")
    return locationdata

def record_game_utter(num_agent, num_prey, save=False): #records game locations AND utterances for a single forward call
    agent = torch.load('models/07-04-2023 1558 medium2312.pt')
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
    if save:
        torch.save((locationdata, utterdata), f"data/utter/{num_agent}{num_prey}utter.pt")
    return locationdata, utterdata

def get_game_metrics(paths, save=False): #for each step in a game in a batch, computes the metrics
    locd, utterdata = torch.load(f'data/utter/{paths[0]}')
    g = torch.flatten(locd,end_dim=1) #we define 2 metrics for a game:
    full = torch.Tensor(len(paths),g.size()[0],2)               #-mean of distance between agents
    for ind, path in enumerate(paths):                          #-mean of distance to the closest prey for each agent
        num_agent = int(path[0])
        num_prey  = int(path[1])
        gamed, utterd = torch.load(f'data/utter/{path}')
        gamedata = torch.flatten(gamed,end_dim=1)
        metric = torch.Tensor(gamedata.size()[0],2)
        for i in range(gamedata.size()[0]):
            agents = gamedata[i,:num_agent,:]
            metric[i,0] = torch.mean(torch.min(torch.cdist(agents,gamedata[i,num_agent:,:],1),1)[0])
            metric[i,1] = torch.mean(torch.cdist(agents,agents,1))
        full[ind] = metric
    result = full.flatten(end_dim=1)
    if save:
        torch.save(result, f"data/utter/metricm.pt")
    return result

def kcluster(path, num_clusters, save=False): #clusters game metrics
    file = torch.load(f'data/utter/{path}')
    device = torch.device('cpu')
    result = kmeans(
        X=file, num_clusters=num_clusters, distance='euclidean', device=device
    )
    if save:
        torch.save(result, f'data/utter/clustermetricm.pt')
    return result

def visualize_clusters(path): #visualizes clustered game metrics
    f1 = torch.load(f'data/utter/{path}')
    f2 = torch.load(f'data/utter/cluster{path}')
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
    visualize_clusters(f'metric{num_agent}{num_prey}.pt')

def visualize_multiple(paths): #visualizes multiple clustered game metrics
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

def plot_losses(paths): #plots losses for each epoch from a file
    fig, ax = plt.subplots()
    for path in paths:
        data = np.load(f'lossdata/{path}.npy')
        min_agents = int(path[0])
        max_agents = int(path[1])
        min_preys = int(path[2])
        max_preys = int(path[3])
        num_epochs = int(path[4:7])

        for a in range(min_agents, max_agents + 1):
            for l in range(min_preys, max_preys + 1):
                plt.plot(np.arange(num_epochs), data[a-1][l-1], label=f"{path}: num_agents: {a} num_prey: {l}")

    ticks = np.arange(0,num_epochs+1,num_epochs/10)
    plt.legend(loc="upper left")
    #ax.set_yscale('log')
    plt.grid()
    ax.set_xticks(ticks)
    plt.show()

def predict_cluster(pathc, pathg): #for a game location data file, computes the assigned cluster for each gamestep in the data
    metrics = get_game_metrics([pathg])
    _, cluster_centers = torch.load(pathc)
    prediction = kmeans_predict(metrics, cluster_centers)
    return pathg, prediction

def complete_data_set():
    locd, utterd = torch.load(f'data/utter/21utter.pt')
    metric = torch.load(f'data/utter/metricm.pt')[:locd.size()[0]*locd.size()[1],:]
    return torch.flatten(locd, end_dim=1), torch.flatten(utterd, end_dim=1), metric
    
#visualize_clusters('metricm.pt')
plot_losses(['2312100new'])
#record_game_utter(2,1,save=True)
#record_game_utter(3,2,save=True)
#get_game_metrics(['21utter.pt', '22utter.pt', '31utter.pt', '32utter.pt'],save=True)
#x,y,z = complete_data_set()
#print(x.size(),y.size(),z.size())
#visualize_multiple(['metric21.pt', 'metric22.pt', 'metric31.pt', 'metric32.pt'])
#kcluster('metricm.pt',7,save=True)
#visualize_clusters('metricm.pt')
#x = torch.load(f'data/utter/21utter.pt')
#print(x.size())