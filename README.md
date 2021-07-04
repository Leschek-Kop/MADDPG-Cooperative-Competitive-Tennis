# Deep Reinforcement Learning Nanodegree - Collaboration and Competition Project 3
#### Solution by Leschek Kopczynski

Third hands-on project of Udacity's Deep Reinforcement Learning Nanodegree. The Solution is presented in Jupyter Notebook wich is running in a conda virtual environment.
This repository contains material related to Udacity's
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
program. Content and text elements to setup the environment is borrowed from Udacity's 
[repository](https://github.com/udacity/deep-reinforcement-learning) and especially from
[Project 3](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

As mentioned as a tip by the course leader the code is oriented by the solutions teached during the drl - nanodegree. 

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. First of all (if you haven't already) install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your System.

2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
cd python
pip install .
```
This will install the dependencies to the conda environment.

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

6. Start Jupyter Notebook Server from (project) root folder with:
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

7. Follow the instructions in the Terminal and navigate to the project by following the provided hyperlink.
8. Once you open the provided link, you have to navigate to `Tennis.ipynb`. Then choose initial the new kernel ( navigate to ):
```bash
kernel -> change kernel -> drlnd
```

#### Congratulations, your setup is ready to go!

## Project and Environment
**NOTE: A full description is provided inside the jupyter notebook**

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### Distributed Training

For this project, agents architecture of [Project 2](https://github.com/Leschek-Kop/DDPG-Continuous-Control-Reacher) is used and modified (code in `ddpg_agent.py`). In `maddpg_agents.py` a wrapper as a multi agent handler is defined to solve the multi agent reinforcement learning task (MARL).
Traning and monitoring is defined in `Tennis.ipynb`.

### Solving the Environment
#### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the GitHub repository, in the `root` folder, and unzip (or decompress) the file. 

#### Instructions
Follow the instructions in `Tennis.ipynb` to get started with training your own agent! 
Note that the task is episodic. Per definition, the agents must get an average score of +0.5 over 100 consecutive episodes
in order to solve the environment.

Results, performance and future work are shown and discussed in `Report.pdf`.