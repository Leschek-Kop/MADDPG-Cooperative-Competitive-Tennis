import numpy as np
import torch
from ddpg_agent import ReplayBuffer
from ddpg_agent import Agent as DDPGAgent


class MultiAgents():
    """Wrapper to handle multiple angents from type 'ddpg_agent.DDPGAgent' """

    def __init__(self, ddpg_config_list, maddpg_config):
        """Initialize an MulriAgents object.

        Params
        ======
            ddpg_config_list (dict): Dict with information about memory mode and agents state and action size
            maddpg_config (list): List with configuration obekts to build agents from type 'ddpg_agent.DDPGAgent'
        """
        self.memory = []
        """ Set a memory mode to handle replay buffer """
        if 'MEMORY_MODE' in maddpg_config:
            if maddpg_config['MEMORY_MODE'] in [0, 1, 2]:
                """ MEMORY_MODE = 0: DDPG - Agents will create own 'Replay Buffer' by using agents configuration """
                # 0- create own via agent config, 1- Use MEMORY from maddpg_config for each agent, 2- Use MEMORY from maddpg_config, one memory shared
                if maddpg_config['MEMORY_MODE'] == 1:
                    """ MEMORY_MODE = 1: DDPG - Agents will use different 'Replay Buffer' created by 'maddpg_config' """
                    for idx, agents_conf in enumerate(ddpg_config_list):
                        self.memory.append(ReplayBuffer(
                            maddpg_config['MEMORY'][idx]['action_size'],
                           maddpg_config['MEMORY'][idx]['buffer_size'],
                           maddpg_config['MEMORY'][idx]['batch_size'],
                           maddpg_config['MEMORY'][idx]['seed'])
                        )
                        agents_conf['kwargs']['MEMORY'] = self.memory[-1]
                elif maddpg_config['MEMORY_MODE'] == 2:
                    """ MEMORY_MODE = 2: DDPG - Agents will share a 'Replay Buffer' created by 'maddpg_config'. Only the 
                    first element in 'maddpg_config['MEMORY']' will be taking into account """
                    if len(self.memory) == 0:
                        self.memory.append(ReplayBuffer(
                            maddpg_config['MEMORY'][0]['action_size'],
                            maddpg_config['MEMORY'][0]['buffer_size'],
                            maddpg_config['MEMORY'][0]['batch_size'],
                            maddpg_config['MEMORY'][0]['seed'])
                        )
                    for agents_conf in ddpg_config_list:
                        agents_conf['kwargs']['MEMORY'] = self.memory[-1]
            else:
                raise Exception("Unknown MEMORY_MODE for MultiAgent configuration")
        else:
            raise Exception("Missing key 'MEMORY_MODE' in MultiAgent configuration")

        self.ddpg_config_list = ddpg_config_list
        self.maddpg_config = maddpg_config
        self.maddpg_agent = [
            DDPGAgent(state_size=conf['state_size'], action_size=conf['action_size'], seed=conf['seed'],
                      hidden_layers_actor=conf['hidden_layers_actor'], hidden_layers_critic=conf['hidden_layers_critic'],
                      **conf['kwargs']) for conf in self.ddpg_config_list]

    def get_agents(self):
        """get all the agents in the MADDPG object"""
        agents = [ddpg_agent for ddpg_agent in self.maddpg_agent]
        return agents

    def reset(self):
        """ Reset exploration - exploitation process for each Agent. """
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()

    def act(self, states, add_noise=True):
        """Returns concatinated actions for given state as per current policy as a list from each agent."""
        state = np.reshape(states, (1, self.maddpg_config['STATE_SIZE']))
        actions = np.array([ddpg_agent.act(state, add_noise) for ddpg_agent in self.maddpg_agent])
        return np.reshape(actions, (1, self.maddpg_config['ACTION_SIZE']*self.maddpg_config['NUM_AGENTS']))

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn for each Agent."""
        states = np.reshape(states, (1, self.maddpg_config['STATE_SIZE']))
        next_states = np.reshape(next_states, (1, self.maddpg_config['STATE_SIZE']))
        action = np.reshape(actions, (self.maddpg_config['NUM_AGENTS'], self.maddpg_config['ACTION_SIZE']))
        for idx, ddpg_agent in enumerate(self.maddpg_agent):
            ddpg_agent.step(states, np.reshape(action[idx], (1, self.maddpg_config['ACTION_SIZE'])), rewards[idx],
                            next_states, dones[idx])

    def save_chkpoints(self, chkpoint_name):
        """ Save model weights form each Agent. """
        for idx, ddpg_agent in enumerate(self.maddpg_agent):
            torch.save(ddpg_agent.actor_local.state_dict(), "{}_{}_{}_model.pth".format(chkpoint_name, 'actor', idx+1))
            torch.save(ddpg_agent.critic_local.state_dict(), "{}_{}_{}_model.pth".format(chkpoint_name, 'critic', idx+1))
