import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode: int, rewards: list, losses: dict):
        total_reward = sum(rewards)
        self.log_scalar("episode/total_reward", total_reward, episode)
        if "actor" in losses:
            self.log_scalar("loss/actor", losses["actor"], episode)
        if "critic" in losses:
            self.log_scalar("loss/critic", losses["critic"], episode)

    def close(self):
        self.writer.close()
