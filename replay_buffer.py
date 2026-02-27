import numpy as np
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))

def is_valid_transition(state, next_state, action, reward):
    for name, state_dict in [('state', state), ('next_state', next_state)]:
        if np.any(np.isnan(state_dict['state'])) or np.any(np.isinf(state_dict['state'])):
            print(f"[ReplayBuffer] Invalid transition: {name}['state'] contains NaN or Inf.")
            return False
        
        if np.any(np.isnan(state_dict['mask'])) or np.any(np.isinf(state_dict['mask'])):
            print(f"[ReplayBuffer] Invalid transition: {name}['mask'] contains NaN or Inf.")
            return False
        
        if np.any(np.isnan(state_dict['control_edges'])) or np.any(np.isinf(state_dict['control_edges'])):
            print(f"[ReplayBuffer] Invalid transition: {name}['control_edges'] contains NaN or Inf.")
            return False

        if np.any(np.isnan(state_dict['control_nodes'])) or np.any(np.isinf(state_dict['control_nodes'])):
            print(f"[ReplayBuffer] Invalid transition: {name}['control_nodes'] contains NaN or Inf.")
            return False

    if np.isnan(reward) or np.isinf(reward):
        print("[ReplayBuffer] Invalid transition: reward contains NaN or Inf.")
        return False
    if not isinstance(action, int) or action < 0:
        print(f"[ReplayBuffer] Invalid transition: action {action} is not a valid int >= 0.")
        return False
    
    return True

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        r"""
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

    def export_state(self):
        return {
            "tree": self.tree,
            "data": self.data,
            "write": self.write,
            "size": self.size,
        }

    def load_state(self, state):
        self.capacity = state["capacity"]
        self.tree = state["tree"]
        self.data = state["data"]
        self.write = state["write"]
        self.size = state["size"]


class ReplayBuffer:
    def __init__(self, capacity, alpha, beta, beta_increment_per_sampling, n_step, gamma):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha  # controls how much prioritization is used
        self.beta = beta  # controls the amount of importance-sampling correction
        self.increment = beta_increment_per_sampling
        self.n_step = n_step  # Multi-Step Learning
        self.gamma = gamma
        self.n_step_buffer = []

    def export_state(self):
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "increment": self.increment,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "n_step_buffer": self.n_step_buffer,
            "sumtree": self.tree.export_state(),
        }

    def load_state(self, state):
        self.capacity = state["capacity"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.increment = state["increment"]
        self.n_step = state["n_step"]
        self.gamma = state["gamma"]
        self.n_step_buffer = state["n_step_buffer"]

        self.tree = SumTree(self.capacity)
        self.tree.load_state(state["sumtree"])

    def add(self, trans):
        self.n_step_buffer.append(trans)

        if len(self.n_step_buffer) == self.n_step:
            # calculate n-step reward
            reward = 0
            for i in range(self.n_step):
                reward += self.gamma**i * self.n_step_buffer[i].reward
                if self.n_step_buffer[i].done:
                    break
            state = self.n_step_buffer[0].state
            action = self.n_step_buffer[0].action
            next_state = self.n_step_buffer[i].next_state
            done = self.n_step_buffer[i].done

            max_priority = np.max(self.tree.tree[-self.capacity :])
            if not np.isfinite(max_priority) or max_priority <= 0:
                max_priority = 10.0
            trans = Transition(state, next_state, action, reward, done)
            self.tree.add(max_priority, trans)
            self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        idxs = []
        priorities = []
        batch = []
        total_p = self.tree.total_priority()
        if not np.isfinite(total_p) or total_p <= 0:
            raise ValueError(f"SumTree total_priority invalid: {total_p}")
        segment = total_p / batch_size
        for i in range(batch_size):
            while True:
                s = np.random.uniform(segment * i, segment * (i + 1))
                idx, priority, data = self.tree.get_leaf(s)
                if isinstance(data, Transition):
                    idxs.append(idx)
                    priorities.append(priority)
                    batch.append(data)
                    break

        batch_states = {
            'state': torch.tensor(np.array([t.state['state'] for t in batch]), dtype=torch.float).to(self.device),
            'mask': torch.tensor(np.array([t.state['mask'] for t in batch]), dtype=torch.float).to(self.device),
            'controls_array': torch.tensor(np.array([t.state['control_edges'] for t in batch]), dtype=torch.float).to(self.device)
        }
        
        batch_states_next = {
            'state': torch.tensor(np.array([t.next_state['state'] for t in batch]), dtype=torch.float).to(self.device),
            'mask': torch.tensor(np.array([t.next_state['mask'] for t in batch]), dtype=torch.float).to(self.device),
            'controls_array': torch.tensor(np.array([t.next_state['control_edges'] for t in batch]), dtype=torch.float).to(self.device)
        }
        actions = torch.tensor(np.array([t.action for t in batch]), dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array([t.reward for t in batch]), dtype=torch.float).unsqueeze(1).to(self.device)
        dones = torch.tensor(np.array([t.done for t in batch]), dtype=torch.float).unsqueeze(1).to(self.device)

        self.beta = np.min([1.0, self.beta + self.increment])
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weight = np.power(self.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = torch.tensor(is_weight, dtype=torch.float).unsqueeze(1).to(self.device)

        return batch_states, actions, rewards, batch_states_next, dones, idxs, is_weight

    def update_priorities(self, idxs, td_errors):
        priorities = (np.abs(td_errors) + 1e-5) ** self.alpha
        priorities = np.minimum(priorities, 10.0)
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority.item())

    def __len__(self):
        return self.tree.size