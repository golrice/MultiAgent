import numpy as np
import torch
import os
from maddpg import MADDPG

class Agent:
    def __init__(self, agent_id, args):
        """
        args: {high_action, action_shape, lr_actor, lr_critic, scenario_name, save_dir, obs_shape}
        """
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor(inputs).squeeze(0)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape) # 高斯噪声
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()
    
    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)