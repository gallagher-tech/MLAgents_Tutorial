Unity ML Example using ML-Agents package
========================================

Guide to setting up an example project (scene itself is premade) and training and running an ML agent in it.

MLAgentsTutorial folder is an empty project with the premade scene, within which this tutorial can be carried out by you.
MLAgentsTutorial_complete folder is the completed version of this tutorial for reference.

What is ML-Agents?
------------------

The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents. PyTorch-based implementation of state-of-the-art algorithms

Installation
------------

1. Have python 3.6+ installed
2. Create python venv in project root dir
3. Install pytorch python pkg https://pytorch.org/get-started/locally/
4. Install ML Agents python pkg: `pip install mlagents` (this will install the latest version listed in the latest release [here](https://github.com/Unity-Technologies/ml-agents/releases))
	- Use `mlagents-learn --help` to confirm dl was successful
5. Open Unity example project
6. Install ML-Agents package (Window > Package Manager)

In this example there is an agent (blue cube guy) and a goal (sphere). Goal is to get the agent to move toward the goal and not fall off the map.

The agent is what's going to be running the AI both for training and then for playing.

Turning a Unity gameobject into an agent involves simply attaching a script to it.

Credits
-------

Based on: https://www.youtube.com/watch?v=zPFU30tbyKs