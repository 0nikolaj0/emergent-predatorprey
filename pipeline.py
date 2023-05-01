import code
import torch
import configs
import numpy as np
import torch.nn as nn
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

from modules.game              import GameModule
from torchmetrics.functional   import kl_divergence
from kmeans_pytorch            import kmeans, kmeans_predict

def record_game_utter(num_agent, num_prey, model): #records game locations AND utterances for a single forward call
    agent = model
    agent.reset()
    agent.train(True)

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

def visualize_clusters(metrics, clusters, similar_points, loc1, loc2, num_agent1, num_agent2, utter): #visualizes clustered game metrics
    cluster_ids_x_und, cluster_centers_und = clusters
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers = cluster_centers_und.detach()
    v = cm.get_cmap('plasma',7)
    x = metrics.detach()
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(235)
    ax4 = fig.add_subplot(233)
    ax1.scatter(x[:, 0], x[:, 1], c=cluster_ids_x,cmap=v)
    ax1.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    ax1.scatter(
        similar_points[:, 0], similar_points[:, 1],
        c='green',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    ax2.scatter(loc1[:num_agent1,0], loc1[:num_agent1,1], c="red")
    ax2.scatter(loc1[num_agent1:,0], loc1[num_agent1:,1], c="blue")
    ax2.set_xlim([-3.5,20])
    ax2.set_ylim([-3.5,20])
    ax3.scatter(loc2[:num_agent2,0], loc2[:num_agent2,1], c="red")
    ax3.scatter(loc2[num_agent2:,0], loc2[num_agent2:,1], c="blue")
    ax3.set_xlim([-3.5,20])
    ax3.set_ylim([-3.5,20])
    start = 0
    for i,val in enumerate(utter):
        ax4.bar([x for x in range(start,start+20)], val, color=[v(i) for k in range(20)])
        start += 20
    ax4.set_xticks(np.arange(0,141,20))
    ax1.axis([torch.min(x[:,0])-1, torch.max(x[:,0])+1, torch.min(x[:,1])-1, torch.max(x[:,1])+1])
    ax1.set_xlabel("mean distance agent to prey")
    ax1.set_ylabel("mean distance between agents")
    plt.tight_layout()
    plt.show()

def pipeline(model, num_agent, num_prey, num_cluster):
    locd, utterd = record_game_utter(num_agent, num_prey, model)
    metrics = get_game_metrics(locd, num_agent)
    clusters = kcluster(metrics, num_cluster)
    #visualize_clusters(metrics, clusters)
    return torch.flatten(locd,end_dim=1).detach(), torch.flatten(utterd, end_dim=1).detach(), metrics.detach(), clusters

def two_datas():
    locd1, utterd1, metrics1, clusters1 = pipeline(torch.load('models/2324300.pt'),3,2,7)
    locd2, utterd2, metrics2, clusters2 = pipeline(torch.load('models/2324300.pt'),3,2,7)
    cluster_ids_x_und, cluster_centers_und = clusters2
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers = cluster_centers_und.detach()
    similar_points = np.zeros((len(cluster_centers)*2,2))
    similar_utter =  []
    for i, val in enumerate(cluster_centers):
        closest_ind1 = torch.argmin(torch.cdist(metrics1,torch.unsqueeze(val,0),1))
        closestm1, closestloc1, closestutt1 = metrics1[closest_ind1], locd1[closest_ind1], utterd1[closest_ind1]
        closest_ind2 = torch.argmin(torch.cdist(metrics2,torch.unsqueeze(closestm1,0),1))
        closestm2, closestloc2, closestutt2 = metrics2[closest_ind2], locd2[closest_ind2], utterd2[closest_ind2]
        similar_points[i] = closestm1
        similar_points[i+len(cluster_centers)] = closestm2
        #similar_utter.append([closestutt1, closestutt2])
    mean_utter = torch.zeros(7,20)
    counter = np.zeros(7)
    for val in cluster_ids_x:
        counter[val] += 1
    for i, ind in enumerate(cluster_ids_x):
        mean_utter[ind] += torch.sum(torch.nan_to_num(utterd2[i],nan=0.0),0) 
    for i, val in enumerate(mean_utter):
        for k, newv in enumerate(val):
            mean_utter[i][k] = newv / counter[i] / 3
    #print(torch.sum(mean_utter,1))
    visualize_clusters(metrics2, clusters2, similar_points, closestloc1, closestloc2, 3, 3, mean_utter)

two_datas()

def plot_losses(paths): #plots losses for each epoch from a file
    fig, ax = plt.subplots()
    for path in paths:
        data = np.load(f'trainingdata/{path}.npy')
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
    ax.set_yscale('log')
    plt.grid()
    ax.set_xticks(ticks)
    plt.show()


#plot_losses(['2222100'])

def predict_cluster(pathc, pathg): #for a game location data file, computes the assigned cluster for each gamestep in the data
    metrics = get_game_metrics([pathg])
    _, cluster_centers = torch.load(pathc)
    prediction = kmeans_predict(metrics, cluster_centers)
    return pathg, prediction