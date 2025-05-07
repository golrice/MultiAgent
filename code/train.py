import numpy as np
from arguments import get_args
from discreteAgents.agent import Agent
from envs.make_env import MultiAgentEnvironment
# from utils.logger import Logger
from utils.seeding import set_seed

"""
The project written below is merely the training process of MADDPG under simple adversary. 
The training methods in other environments are similar
"""

manual_seed = 42

set_seed(manual_seed)
args = get_args()

#logger = Logger(config["log_dir"])

env = MultiAgentEnvironment("simple_adversary", "human")

agents = {}
obs_spaces = {}
action_spaces = {}
for agent in env.possible_agents:
    obs_spaces[agent] = np.array(env.observation_space(agent).shape).flatten()[0]
    action_spaces[agent] = env.action_space(agent).n

for agent in env.possible_agents:
    agents[agent] = Agent(agent, obs_spaces[agent], action_spaces[agent], sum(obs_spaces.values()), args.lr_actor, args.lr_critic, args.tau, args.gamma)

obs = env.reset(manual_seed)

epsilon = 0.9
for step in range(args.max_episode_len):
    actions = {}  
    for agent in env.env.agents:
        actions[agent] = agents[agent].select_action(obs[agent], epsilon)
    
    print(f"actions = {actions}")
    next_obs, rewards, dones, infos = env.step(actions)

    # buffer.store(obs, actions, rewards, next_obs, dones)
    # train_policy()

    obs = next_obs
    if all(dones.values()):
        obs = env.reset()
        exit(0)

# logger.log_scalar(tag="learning_rate", value=0.001, step=100)
# logger.close()
