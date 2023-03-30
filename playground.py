import torch
from modules.game import GameModule
from configs import default_game_config, get_game_config
import code


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

agent = torch.load('models/29-03-2023 1622 easy2111.pt')
agent.reset()
agent.train(False)
game = GameModule(default_game_config, 4, 1)
agent(game)
#code.interact(local=locals())
