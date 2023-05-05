import code
import torch
import configs
import numpy as np
import torch.nn as nn
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import random
import itertools

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

def generate_color_list(length, start_color, end_color):
    start_rgb = tuple(int(start_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    end_rgb = tuple(int(end_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    step_size = [(end_rgb[i] - start_rgb[i]) / (length-1) for i in range(3)]
    colors = []
    for i in range(length):
        color = tuple(round(start_rgb[j] + step_size[j] * i) for j in range(3))
        colors.append('#{:02x}{:02x}{:02x}'.format(*color))
    return colors

def random_color():
    colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845', '#00308F', '#0074D9', '#2ECC40', '#FFDC00', '#FF851B', '#FF4136', '#85144b', '#111111', '#AAAAAA', '#DDDDDD', '#001f3f', '#0074D9', '#7FDBFF', '#39CCCC', '#3D9970', '#2ECC40', '#01FF70', '#FF851B', '#FF4136', '#F012BE', '#B10DC9', '#85144b', '#111111', '#AAAAAA', '#DDDDDD']
    return random.choice(colors)

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

def visualize_clusters(metrics1, clusters1, metrics2, clusters2, similar_points, utter, utter2): #visualizes clustered game metrics
    cluster_ids_x_und, cluster_centers_und1 = clusters1
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers1 = cluster_centers_und1.detach()
    v = cm.get_cmap('plasma',7)
    x = metrics1.detach()
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(234)
    ax4 = fig.add_subplot(232)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(133)
    ax1.scatter(x[:, 0], x[:, 1], c=cluster_ids_x,cmap=v)
    ax1.scatter(
        cluster_centers1[:, 0], cluster_centers1[:, 1],
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

    cluster_ids_x_und, cluster_centers_und2 = clusters2
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers2 = cluster_centers_und2.detach()
    v = cm.get_cmap('plasma',7)
    x = metrics2.detach()
    ax2.scatter(x[:, 0], x[:, 1], c=cluster_ids_x,cmap=v)
    ax2.scatter(
        cluster_centers2[:, 0], cluster_centers2[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    ax2.scatter(
        similar_points[:, 0], similar_points[:, 1],
        c='green',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    map = torch.argmin(torch.cdist(cluster_centers1, cluster_centers2,1),0)
    kl_div = torch.Tensor(7)
    for i, val in enumerate(map):
        p = torch.unsqueeze(utter2[i], 0)
        q = torch.unsqueeze(utter2[val], 0)
        res = kl_divergence(p, q)
        kl_div[i] = res

    ax6.bar([x for x in range(len(kl_div))], kl_div)

    start = 0
    for i,val in enumerate(utter):
        ax4.bar([x for x in range(start,start+20)], val, color=[v(i) for k in range(20)])
        start += 20
    ax4.set_xticks(np.arange(0,141,20))
    start = 0
    for i,val in enumerate(utter2):
        ax5.bar([x for x in range(start,start+20)], val, color=[v(i) for k in range(20)])
        start += 20
    ax5.set_xticks(np.arange(0,141,20))
    ax1.axis([torch.min(x[:,0])-1, torch.max(x[:,0])+1, torch.min(x[:,1])-1, torch.max(x[:,1])+1])
    ax1.set_xlabel("mean distance agent to prey")
    ax1.set_ylabel("mean distance between agents")
    plt.tight_layout()
    plt.show()



def pipeline(model, num_agent, num_prey, num_cluster):
    locd, utterd = record_game_utter(num_agent, num_prey, model)
    metrics = get_game_metrics(locd, num_agent)
    clusters = kcluster(metrics, num_cluster)
    return torch.flatten(locd,end_dim=1).detach(), torch.flatten(utterd, end_dim=1).detach(), metrics.detach(), clusters

def two_datas():
    locd1, utterd1, metrics1, clusters1 = pipeline(torch.load('models/3423100from.pt'),3,4,7)
    locd2, utterd2, metrics2, clusters2 = pipeline(torch.load('models/3423100.pt'),3,4,7)
    cluster_ids_x_und, cluster_centers_und = clusters1
    cluster_ids_x = cluster_ids_x_und.detach()
    cluster_centers = cluster_centers_und.detach()
    similar_points = np.zeros((len(cluster_centers)*2,2))
    cluster_ids_x_und2, cluster_centers_und2 = clusters2
    cluster_ids_x2 = cluster_ids_x_und2.detach()
    cluster_centers2 = cluster_centers_und2.detach()

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
        mean_utter[ind] += utterd1[i][0]
    for i, val in enumerate(mean_utter):
        for k, newv in enumerate(val):
            mean_utter[i][k] = newv / counter[i]
    mean_utter2 = torch.zeros(7,20)
    counter2 = np.zeros(7)
    for val in cluster_ids_x2:
        counter2[val] += 1
    for i, ind in enumerate(cluster_ids_x2):
        mean_utter2[ind] += torch.sum(utterd2[i],0)
    for i, val in enumerate(mean_utter2):
        for k, newv in enumerate(val):
            mean_utter2[i][k] = newv / counter2[i] / 3
    visualize_clusters(metrics1, clusters1, metrics2, clusters2, similar_points, mean_utter, mean_utter2)

two_datas()

def get_plot1(paths):
    num_agent = 2
    num_prey = 1
    fig = plt.figure(figsize=(8,6))
    v = cm.get_cmap('plasma',7)
    for i, path in enumerate(paths):
        locd, utterd, metrics, clusters = pipeline(torch.load(path), num_agent, num_prey,7)
        cluster_ids_x_und, cluster_centers_und = clusters
        cluster_ids_x = cluster_ids_x_und.detach()
        cluster_centers = cluster_centers_und.detach()
        x = metrics.detach()
        ax1 = fig.add_subplot(len(paths),3,i*3+1)
        ax2 = fig.add_subplot(len(paths),3,i*3+2)
        ax3 = fig.add_subplot(len(paths),3,i*3+3)
        ax1.scatter(x[:, 0], x[:, 1], c=cluster_ids_x,cmap=v)
        ax1.scatter(
            cluster_centers[:, 0], cluster_centers[:, 1],
            c='white',
            alpha=0.6,
            edgecolors='black',
            linewidths=2)
        ax1.set_title(f'num_agent: {num_agent}, num_prey: {num_prey}')
        ax1.set_xlabel("mean distance agent to prey")
        ax1.set_ylabel("mean distance between agents")

        ax2.set_title(f'model: {path}')
        ax2.scatter(locd[0,:num_agent,0], locd[0,:num_agent,1], c="red")
        ax2.scatter(locd[0,num_agent:,0], locd[0,num_agent:,1], c="blue")
        ax2.set_xlim([-3.5,20])
        ax2.set_ylim([-3.5,20])

        utter = torch.zeros(7,20)
        counter = np.zeros(7) 
        for val in cluster_ids_x:
            counter[val] += 1
        for i, ind in enumerate(cluster_ids_x):
            utter[ind] += utterd[i][0]
        for i, val in enumerate(utter):
            for k, newv in enumerate(val):
                utter[i][k] = newv / counter[i]

        start = 0
        for i,val in enumerate(utter):
            ax3.bar([x for x in range(start,start+20)], val, color=[v(i) for _ in range(20)])
            start += 20
        ax3.set_xticks(np.arange(0,141,20))
        num_agent += 1
        num_prey  += 1
    plt.tight_layout()
    plt.show()

#get_plot1(['models/2311100.pt', 'models/2322100noload.pt','models/3423100noload.pt'])

def plot_losses(paths): #plots losses for each epoch from a file
    fig, ax = plt.subplots()
    for path in paths:
        data = np.load(f'trainingdata/{path}.npy')
        colors = generate_color_list(20, random_color(), random_color())
        min_agents = int(path[0])
        max_agents = int(path[1])
        min_preys = int(path[2])
        max_preys = int(path[3])
        num_epochs = int(path[4:7])
        marker = itertools.cycle((',', '+', '.', 'o', '*')) 
        counter = 0
        for a in range(min_agents, max_agents + 1):
            for l in range(min_preys, max_preys + 1):
                plt.plot(np.arange(num_epochs), data[a-1][l-1], label=f"{path}: num_agents: {a} num_prey: {l}", color=colors[counter], marker=next(marker))
                counter += 1

    ticks = np.arange(0,num_epochs+1,num_epochs/10)
    plt.legend(loc="upper right")
    #ax.set_yscale('log')
    ax.set_ylim(1,100)
    ax.set_yticks([1,2,4,8,16,32])
    #ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.grid()
    ax.set_xticks(ticks)
    plt.show()


#plot_losses(['3423100distances','3423100distancesnoutter'])
#plot_losses(['2322100distancesnoload'])

def predict_cluster(pathc, pathg): #for a game location data file, computes the assigned cluster for each gamestep in the data
    metrics = get_game_metrics([pathg])
    _, cluster_centers = torch.load(pathc)
    prediction = kmeans_predict(metrics, cluster_centers)
    return pathg, prediction

def utter3(path):
    utter = torch.load(path).detach()
    flattened = torch.flatten(utter, start_dim=1, end_dim=2)
    num_epochs = flattened.size()[0]
    data_size = flattened.size()[1]
    colors = generate_color_list(100, '#194526', '#d63131')
    for i in range(num_epochs):
        y = flattened[i]
        x = torch.sum(y,0) / data_size
        plt.scatter([i for _ in range(20)], [o for o in range(20)], s=x*200, c=colors[i % len(colors)])
    plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False)
    plt.yticks([x for x in range(20)])
    plt.xlabel('training epoch')
    plt.ylabel('vocabulary symbol usage')
    plt.show()

#utter3('trainingdata/utter3423100.pt')


