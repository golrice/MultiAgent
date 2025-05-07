import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Actor, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
    
    def forward(self, x):
        logits = F.softmax(self.seq(x), dim=-1)
        # probs = torch.argmax(logits)
        return logits

class Critic(nn.Module):
    def __init__(self, in_features):
        super(Critic, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action):
        return self.seq(torch.cat([state, action]), dim=1)
