from tqdm import tqdm
from agent import Agent
from relay_buffer import Buffer
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.env = env
        self.episode_limit = args.max_episode_len
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            # 生成npc的动作？
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent) # 去除自己，但是好像训练的时候也注意到去除自己了，所以可能不去可以
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns), returns))
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/returns.png', format='png')
            self.noise = max(0.05, self.noise - 5e-7)
            self.epsilon = max(0.05, self.epsilon - 5e-7)
            np.save(self.save_path + 'returns.npy', returns)

        def evaluate(self):
            returns = []
            for episode in range(self.args.evaluate_episodes):
                # 初始化环境
                s = self.env.reset()
                rewards = 0
                for time_step in range(self.args.evaluate_episode_len):
                    self.env.render()
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                    for i in range(self.args.n_agents, self.args.n_players):
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    s_next, r, done, info = self.env.step(actions)
                    rewards += r[0]
                    s = s_next
                returns.append(rewards)
                print(f'returns is {rewards}')
            return sum(returns) / self.args.evaluate_episodes
