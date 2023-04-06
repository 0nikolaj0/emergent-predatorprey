# emergent-predatorprey
An implementation of Emergence of Grounded Compositional Language in a Multi-Agent Predator-Prey Game

Based on An implementation of Emergence of Grounded Compositional Language in Multi-Agent Populations by Igor Mordatch and Pieter Abbeel

To run, invoke `python train.py` in environment with PyTorch installed. To experiment with parameters, invoke `python train.py --help` to get a list of command line arguments that modify parameters.

* `game.py` provides a non-tensor based implementation of the game mechanics (used for game behavior exploration and random game generation during training
* `modules/agent.py` defines the general module that's responsible for the execution of the overall policy throughout training. It holds all information pertaining to the whole training episode, and at each forward pass runs a given game until the end, returning the total cost all agents collected over the entire game
* `train.py` provides the training harness that runs many games and trains the agents
* `configs.py` provides the data structures that are passed as configuration to various modules in the computational graph as well as the default values used in training now
* `constants.py` provides constant factors that shouldn't need modification during regular running of the model
* `visualize.py` provides a computational graph visualization tool taken from [here](https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py)
* `comp-graph.pdf` is a pdf visualization of the computational graph of the game-agent mechanics
* `collect.py` defines all the methods that I have used to collect data, like the agents' utterances, game data