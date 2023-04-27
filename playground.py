import torch
from modules.game import GameModule
from configs import default_game_config, get_game_config
import code


config = {
        'batch_size': default_game_config.batch_size,
        'world_dim': default_game_config.world_dim,
        'max_agents': default_game_config.max_agents,
        'max_prey': default_game_config.max_prey,
        'min_agents': default_game_config.min_agents,
        'min_prey': default_game_config.min_prey,
        'num_shapes': default_game_config.num_shapes,
        'num_colors': default_game_config.num_colors,
        'no_utterances': not default_game_config.use_utterances,
        'vocab_size': default_game_config.vocab_size,
        'memory_size': default_game_config.memory_size
    }

agent = torch.load('latest.pt')
agent.reset()
agent.train(False)
game = GameModule(default_game_config, 3, 2)
agent.forward(game)
#code.interact(local=locals())
