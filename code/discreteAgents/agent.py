import numpy as np
import torch
from discreteAgents.maddpg import MADDPG

class Agent:
    def __init__(self, name, actor_infeatures, actor_outfeatures, critic_infeatures, lr_actor, lr_critic, tau, gamma):
        self.name = name
        self.actions_shape = actor_outfeatures
        self.policy = MADDPG(actor_infeatures, actor_outfeatures, critic_infeatures, lr_actor, lr_critic, tau, gamma)

    def select_action(self, o, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.choice(self.actions_shape)
            return u
        
        inputs = torch.tensor(o, dtype=torch.float32)
        pi = self.policy.actor(inputs)
        return torch.argmax(pi).item()

    def learn(self, transitions, all_agents):
        self.policy.train(self.name, transitions, all_agents)
