import torch
import os
from model import Actor, Critic
import torch.nn.functional as F

class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # 初始化四个网络
        self.actor = Actor(args, agent_id)
        self.critic = Critic(args)
        self.actor_target = Actor(args, agent_id)
        self.critic_target = Critic(args)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        os.makedirs(self.args.save_dir, exist_ok=True)
        self.model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        os.makedirs(self.model_path, exist_ok=True)
        self.model_path = os.path.join(self.model_path, f'agent_{self.agent_id}')
        os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(os.path.join(self.model_path, 'init_actor.pth')):
            self.actor.load_state_dict(torch.load(os.path.join(self.model_path, 'init_actor.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(self.model_path, 'init_critic.pth')))
            print(f'agent {self.agent_id} loaded')

    def _soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions[f'r_{self.agent_id}'] # 训练时的奖励用的时自己的奖励
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions[f'o_{agent_id}'])
            u.append(transitions[f'u_{agent_id}'])
            o_next.append(transitions[f'o_next_{agent_id}'])
        
        # 计算目标Q值
        u_next = []
        with torch.no_grad():
            index = 0 # 通过index来找到对应的actor_target，能够跳过自己
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target(o_next[agent_id]))
                else:
                    u_next.append(other_agents[index].policy.actor_target(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

            q_value = self.critic(o, u)
            critic_loss = F.mse_loss(q_value, target_q)

        # 重新选择当前agent的动作，其它agent的动作不变
        u[self.agent_id] = self.actor(o[self.agent_id])
        actor_loss = -self.critic(o, u).mean()

        # 更新网络参数
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update()
        if (self.train_step + 1) % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.model_path, f'agent_{self.agent_id}_{num}')
        torch.save(self.actor.state_dict(), model_path + '_actor.pth')
        torch.save(self.critic.state_dict(), model_path + '_critic.pth')
        # print(f'agent {self.agent_id} saved')