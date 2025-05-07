from pettingzoo.mpe import (
    simple_adversary_v3,
    simple_crypto_v3,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
    simple_spread_v3,
    simple_tag_v3,
    simple_world_comm_v3
)
import numpy as np

class MultiAgentEnvironment:
    def __init__(self, env_name: str, render_mode: str = None):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = self._create_env()
        self.possible_agents = self.env.possible_agents
        self.num_agents = len(self.possible_agents)

    def _create_env(self):
        env_mapping = {
            "simple_adversary": simple_adversary_v3,
            "simple_crypto": simple_crypto_v3,
            "simple_push": simple_push_v3,
            "simple_reference": simple_reference_v3,
            "simple_speaker_listener": simple_speaker_listener_v4,
            "simple_spread": simple_spread_v3,
            "simple_tag": simple_tag_v3,
            "simple_world_comm": simple_world_comm_v3
        }
        
        if self.env_name not in env_mapping:
            raise ValueError(
                f"Environment '{self.env_name}' is not supported. "
                f"Available environments: {list(env_mapping.keys())}"
            )
        
        env_module = env_mapping[self.env_name]
        return env_module.parallel_env(render_mode=self.render_mode)

    def reset(self, seed: int = None):
        observations, infos = self.env.reset(seed=seed)
        return observations

    def step(self, actions: dict):
        """
        actions: dict {agent_name: action_tensor}
        """
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = {agent: terminations[agent] or truncations[agent] for agent in self.env.agents}
        return obs, rewards, dones, infos

    def get_state(self):
        if hasattr(self.env, "state"):
            return self.env.state()
        else:
            obs_dict = self.env.observe()
            return np.concatenate([obs_dict[agent] for agent in self.env.agents], axis=0)

    def action_space(self, agent_id):
        return self.env.action_space(agent_id)

    def observation_space(self, agent_id):
        return self.env.observation_space(agent_id)

    def close(self):
        self.env.close()
