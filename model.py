import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, output_size, seed, hidden_layers=None, drop_p=None):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each state
            output_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): Number of nodes corresponding to layers
            drop_p (int): dropout proberbility
        """
        super(QNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        # Add the last layer, hidden layer to a output
        self.output = nn.Linear(hidden_layers[-1], output_size)
        # To Initialize with 0
        # self.hidden_layers[...].bias.data.fill_(0) with zeros
        # self.hidden_layers[...].weight.data.normal_(std=0.01) with normal

        # Dropout
        self.dropout = None
        if drop_p is not None:
            self.dropout = nn.Dropout(p=drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            if self.dropout:
                state = self.dropout(state)

        state = self.output(state)
        return state
        # return F.softmax(state, dim=1)
        # return F.log_softmax(state, dim=1)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=None):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (array): Number of hidden layers and nodes
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        if hidden_layers is None:
            hidden_layers = [256, 128]
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        # Add the last layer, hidden layer to a output
        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initilize nodes for each layer"""
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions. Forward through each layer in `hidden_layers`,
            with ReLU activation
        """
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
        state = self.output(state)
        return torch.tanh(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=None,):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (array): Number of hidden layers and nodes
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        if hidden_layers is None:
            hidden_layers = [256, 128]
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        first_hidden_layer = True
        for h1, h2 in layer_sizes:
            if first_hidden_layer:
                self.hidden_layers.extend([nn.Linear(h1+action_size, h2)])
                first_hidden_layer = False
            else:
                self.hidden_layers.extend([nn.Linear(h1, h2)])
        # Add the last layer, hidden layer to a output
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initilize nodes for each layer"""
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values. Forward through each layer in
        `hidden_layers`, with ReLU activation.
        """

        first_hidden_layer = True
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            if first_hidden_layer:
                state = torch.cat((state, action), dim=1)
                first_hidden_layer = False
        return self.output(state)
