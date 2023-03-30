import torch
import torch.nn as nn

import configs

agent = torch.load('models/29-03-2023 1622 easy2111.pt')
agent.reset()
agent.train(False)

config = configs.default_game_config
