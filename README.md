# emergent-predatorprey
An implementation of Emergence of a performant communication protocol in a multi-agent predator-prey game by Nikolaj Wilms

An extension of An implementation of Emergence of Grounded Compositional Language in Multi-Agent Populations by Igor Mordatch and Pieter Abbeel

To run, invoke `python train.py` in environment with PyTorch installed. To experiment with parameters, invoke `python train.py --help` to get a list of command line arguments that modify parameters. Currently training just prints out the loss of each game episode run, without any further analysis, and the model weights are not saved at the end. These features are coming soon.

* `modules/game.py` provides a non-tensor based implementation of the game mechanics (used for game behavior exploration and random game generation during training
* `modules/agent.py` provides the full computational model including agent and game dynamics through an entire episode
* `models/` provides pre-computed models. The names of these files correspond to their training configuration (name[0] = min_agents, name[1] = max_agents, name[2] = min_prey, name[3] = max_prey, name[4:] = amount of epochs). This folder is also used as the output folder when saving models. 
* `trainingdata/` provides an output folder for training data. The naming convention follows that of the models folder. Each file holds either communication data, loss data, or distance (from the goal) data for a single training session  
* `train.py` provides the training harness that runs many games and trains the agents
* `configs.py` provides the data structures that are passed as configuration to various modules in the computational graph as well as the default values used in training now
* `constants.py` provides constant factors that shouldn't need modification during regular running of the model
* `pipeline.py` provides all data mining and visualization for the plots in the thesis
