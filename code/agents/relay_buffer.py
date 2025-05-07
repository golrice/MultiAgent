import threading
import numpy as np

class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        self.current_size = 0
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer[f'o_{i}'] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer[f'u_{i}'] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer[f'r_{i}'] = np.empty([self.size])
            self.buffer[f'o_next_{i}'] = np.empty([self.size, self.args.obs_shape[i]])

        self.lock = threading.Lock()

    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer[f'o_{i}'][idxs] = o[i]
                self.buffer[f'u_{i}'][idxs] = u[i]
                self.buffer[f'r_{i}'][idxs] = r[i]
                self.buffer[f'o_next_{i}'][idxs] = o_next[i]

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, size=batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer
    
    def _get_storage_idx(self, inc=1):
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, size=overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, size=inc)
        
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
        