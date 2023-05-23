import argparse
import numpy as np
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import configs
from modules.agent import AgentModule
from modules.game import GameModule
from collections import defaultdict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
parser.add_argument('--no-utterances', action='store_true', help='if specified disables the communications channel (default enabled)')
parser.add_argument('--penalize-words', action='store_true', help='if specified penalizes uncommon word usage (default disabled)')
parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
parser.add_argument('--learning-rate', type=float, help='if specified sets learning rate (default 1e-3)')
parser.add_argument('--batch-size', type=int, help='if specified sets batch size(default 256)')
parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default 32)')
parser.add_argument('--num-shapes', '-s', type=int, help='if specified sets number of colors (default 3)')
parser.add_argument('--num-colors', '-c', type=int, help='if specified sets number of shapes (default 3)')
parser.add_argument('--max-agents', type=int, help='if specified sets maximum number of agents in each episode (default 3)')
parser.add_argument('--min-agents', type=int, help='if specified sets minimum number of agents in each episode (default 1)')
parser.add_argument('--max-prey', type=int, help='if specified sets maximum number of preys in each episode (default 3)')
parser.add_argument('--min-prey', type=int, help='if specified sets minimum number of preys in each episode (default 1)')
parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode (default 6)')
parser.add_argument('--world-dim', '-w', type=int, help='if specified sets the side length of the square grid where all agents and preys spawn(default 16)')
parser.add_argument('--oov-prob', '-o', type=int, help='higher value penalize uncommon words less when penalizing words (default 6)')
parser.add_argument('--load-model-weights', type=str, help='if specified start with saved model weights saved at file given by this argument')
parser.add_argument('--save-model-weights', type=str, help='if specified save the model weights at file given by this argument')
parser.add_argument('--visibility', type=int, help='if specified sets the visibility range for all agents')
parser.add_argument('--use_visibility', action='store_true', help='if specified enables visibility range for all agents')
parser.add_argument('--use-cuda', action='store_true', help='if specified enables training on CUDA (default disabled)')

def print_losses(epoch, losses, dists, game_config):
    for a in range(game_config.min_agents, game_config.max_agents + 1):
        for l in range(game_config.min_prey, game_config.max_prey + 1):
            loss = losses[a][l][-1] if len(losses[a][l]) > 0 else 0
            min_loss = min(losses[a][l]) if len(losses[a][l]) > 0 else 0

            dist = dists[a][l][-1] if len(dists[a][l]) > 0 else 0
            min_dist = min(dists[a][l]) if len(dists[a][l]) > 0 else 0

            print("[epoch %d][%d agents, %d preys][%d batches][last loss: %f][min loss: %f][last dist: %f][min dist: %f]" % (epoch, a, l, len(losses[a][l]), loss, min_loss, dist, min_dist))
    print("_________________________")

def main():
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)
    print("Training with config:")
    print(training_config)
    print(game_config)
    print(agent_config)
    agent = AgentModule(agent_config)
    if training_config.use_cuda:
        agent.cuda()
    optimizer = RMSprop(agent.parameters(), lr=training_config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, cooldown=5)
    losses = defaultdict(lambda:defaultdict(list))
    dists = defaultdict(lambda:defaultdict(list))
    l = np.zeros((game_config.max_agents, game_config.max_prey, training_config.num_epochs))
    d = np.zeros((game_config.max_agents, game_config.max_prey, training_config.num_epochs))
    if agent_config.use_utterances:
        u = torch.Tensor(training_config.num_epochs,4,game_config.batch_size,game_config.vocab_size)
    for epoch in range(training_config.num_epochs):
        num_agents = np.random.randint(game_config.min_agents, game_config.max_agents+1)
        num_preys = np.random.randint(game_config.min_prey, game_config.max_prey+1)
        agent.reset()
        game = GameModule(game_config, num_agents, num_preys)
        if training_config.use_cuda:
            game.cuda()
        optimizer.zero_grad()

        total_loss, timesteps = agent(game)
        per_agent_loss = total_loss.data[0] / num_agents / game_config.batch_size
        losses[num_agents][num_preys].append(per_agent_loss)

        l[num_agents-1][num_preys-1][epoch] = per_agent_loss
        for i in range(game_config.max_agents):
            for k in range(game_config.max_prey):
                if (i+1, k+1) != (num_agents,num_preys):
                    l[i][k][epoch] = l[i][k][epoch-1]

        if agent_config.use_utterances:
            for i in range(4):
                u[epoch,i] = torch.sum(timesteps[int(i/4*agent_config.time_horizon)]['utterances'],1) / num_agents

        dist = game.get_avg_agent_to_goal_distance()
        avg_dist = dist.data.item() / num_agents / game_config.batch_size
        dists[num_agents][num_preys].append(avg_dist)

        d[num_agents-1][num_preys-1][epoch] = avg_dist
        for i in range(game_config.max_agents):
            for k in range(game_config.max_prey):
                if (i+1, k+1) != (num_agents,num_preys):
                    d[i][k][epoch] = d[i][k][epoch-1]

        print_losses(epoch, losses, dists, game_config)

        total_loss.backward()
        optimizer.step()

        if num_agents == game_config.max_agents and num_preys == game_config.max_prey:
            scheduler.step(losses[game_config.max_agents][game_config.max_prey][-1])

    if training_config.save_model:
        np.save(f'trainingdatan/{game_config.min_agents}{game_config.max_agents}{game_config.min_prey}{game_config.max_prey}{training_config.num_epochs}', l)
        np.save(f'trainingdatan/{game_config.min_agents}{game_config.max_agents}{game_config.min_prey}{game_config.max_prey}{training_config.num_epochs}distances', d)
        if agent_config.use_utterances:
            torch.save(u, f'trainingdatan/utter{game_config.min_agents}{game_config.max_agents}{game_config.min_prey}{game_config.max_prey}{training_config.num_epochs}.pt')
        torch.save(agent, training_config.save_model_file)
        print("Saved agent model weights at %s" % training_config.save_model_file)
        
    """
    import code
    code.interact(local=locals())
    """


if __name__ == "__main__":
    main()

