import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


def final_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    nn.init.uniform_(layer.bias, -3e-3, 3e-3)


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        q1 = self.linear3(F.relu(self.linear2(F.relu(self.linear1(xu)))))
        q2 = self.linear6(F.relu(self.linear5(F.relu(self.linear4(xu)))))
        return q1, q2


class QHead(nn.Module):

    def __init__(self, num_inputs, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.apply(weights_init_)

    def forward(self, x):
        return self.linear3(F.relu(self.linear2(F.relu(self.linear1(x)))))


class DeterministicCritic(nn.Module):
    """TD3 uses two heads (twin=True)."""

    def __init__(self, num_inputs=48, num_actions=2, hidden_size=256, twin=True):
        super().__init__()
        self.heads = nn.ModuleList(
            [QHead(num_inputs + num_actions, hidden_size) for _ in range(2 if twin else 1)]
        )
        for head in self.heads:
            final_layer_init(head.linear3)

    def forward(self, state, action):
        inputs = torch.cat((state, action), dim=1)
        return tuple(head(inputs) for head in self.heads)
