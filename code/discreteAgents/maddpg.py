import torch
import os
from discreteAgents.model import Actor, Critic
import torch.nn.functional as F

class MADDPG:
    def __init__(self, actor_infeatures, actor_outfeatures, critic_infeatures, lr_actor, lr_critic, tau, gamma):
        self.tau = tau
        self.gamma = gamma

        self.actor = Actor(actor_infeatures, actor_outfeatures)
        self.critic = Critic(critic_infeatures)
        self.actor_target = Actor(actor_infeatures, actor_outfeatures)
        self.critic_target = Critic(critic_infeatures)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def load_save_parameter(self, save_dir, save_rate, scenario_name, agent_id):
        self.save_rate = save_rate
        self.agent_id = agent_id

        os.makedirs(save_dir, exist_ok=True)
        self.model_path = os.path.join(save_dir, scenario_name)
        os.makedirs(self.model_path, exist_ok=True)
        self.model_path = os.path.join(self.model_path, f'agent_{self.agent_id}')
        os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(os.path.join(self.model_path, 'init_actor.pth')):
            self.actor.load_state_dict(torch.load(os.path.join(self.model_path, 'init_actor.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(self.model_path, 'init_critic.pth')))

    def _soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def train(self, name: str, transitions, all_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions[f'r_{self.agent_id}'] # 训练时的奖励用的时自己的奖励
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions[f'o_{agent_id}'])
            u.append(transitions[f'u_{agent_id}'])
            o_next.append(transitions[f'o_next_{agent_id}'])
 
        u_next = []
        with torch.no_grad():
            u_next = [agent.policy.actor_target[o_next[agent.name]] for agent in all_agents]
            q_next = self.critic_target(o_next, u_next).detach()
            target_q = (r.unsqueese(1) + self.gamma * q_next).detach()

            q_value = self.critic(o, u)
            critic_loss = F.mse_loss(q_value, target_q)
        
        u[name] = self.actor(o[name])
        actor_loss = -self.critic(o, u).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.model_path, f'agent_{self.agent_id}_{num}')
        torch.save(self.actor.state_dict(), model_path + '_actor.pth')
        torch.save(self.critic.state_dict(), model_path + '_critic.pth')
        # print(f'agent {self.agent_id} saved')