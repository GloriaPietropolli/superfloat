"""
Implementation of the deep learning routine
"""
import torch
import torch.nn as nn
from preparation_data import input_size

A = 4 / 3
topology = [input_size, 31, 23, 1]


def mysigmoid(x):
    return A * torch.tanh(x)  # A*(np.exp(a*x) -1)/(np.exp(a*x)+1)


class MySigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mysigmoid(x)


activation_function = MySigmoid()


class MLP_Bayesian(nn.Module):
    def __init__(self, top):
        input_siz, n_hidden1, n_hidden2, output_size = topology
        super(MLP_Bayesian, self).__init__()
        self.input_siz = input_siz
        self.network = nn.Sequential(
            nn.Linear(input_siz, n_hidden1),  # BayesianLinear(input_size, n_hidden1),  #
            activation_function,  # nn.SELU(),  #  funzione bene con nn.ReLU(), ELU(),
            nn.Linear(n_hidden1, n_hidden2),  # BayesianLinear(n_hidden1, n_hidden2),  #
            activation_function,  # nn.SELU(), #
            nn.Linear(n_hidden2, output_size),  # BayesianLinear(n_hidden2, output_size),  #
        )

    def forward(self, x):
        x = x.view(-1, self.input_siz)
        return self.network(x)
