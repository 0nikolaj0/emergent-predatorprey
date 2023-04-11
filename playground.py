import torch
from modules.game import GameModule
from configs import default_game_config, get_game_config
import code
import visualize


config = {
        'batch_size': default_game_config.batch_size,
        'world_dim': default_game_config.world_dim,
        'max_agents': default_game_config.max_agents,
        'max_preys': default_game_config.max_preys,
        'min_agents': default_game_config.min_agents,
        'min_preys': default_game_config.min_preys,
        'num_shapes': default_game_config.num_shapes,
        'num_colors': default_game_config.num_colors,
        'no_utterances': not default_game_config.use_utterances,
        'vocab_size': default_game_config.vocab_size,
        'memory_size': default_game_config.memory_size
    }

agent = torch.load('models/10-04-2023 1511 hard3412.pt')
agent.reset()
agent.train(False)
game = GameModule(default_game_config, 4, 2)
agent(game)
#code.interact(local=locals())