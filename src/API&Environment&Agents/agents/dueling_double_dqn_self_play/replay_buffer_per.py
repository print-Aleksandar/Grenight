from collections import deque
import numpy as np


class Transition:
    __slots__ = ["state", "legal_mask", "action", "reward", "next_state", "done", "next_legal_mask"]

    def __init__(self, state, legal_mask, action, reward, next_state, done, next_legal_mask):
        self.state = state
        self.legal_mask = legal_mask
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.next_legal_mask = next_legal_mask


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = deque(maxlen=capacity)
        self.max_priority = 1.0
        self.write_ptr = 0

    def add(self, priority: float, transition: Transition):
        idx = self.write_ptr + self.capacity - 1
        self.data.append(transition)
        self.update(idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority

        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def sample(self, batch_size: int):
        priorities = self.tree[self.capacity - 1:]
        priorities = priorities[:len(self.data)]

        if priorities.sum() == 0:
            return np.random.choice(len(self.data), size=batch_size, replace=False)

        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.data), size=batch_size, p=probs, replace=False)
        return indices, probs[indices]

    def __len__(self):
        return len(self.data)


class ReplayBufferPER:

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, beta_steps: int = 1_000_000):

        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.step_count = 0
        self.epsilon = 1e-6

    def push(self, *args):
        transition = Transition(*args)
        priority = self.tree.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int):
        if len(self.tree) < batch_size:
            indices = list(range(len(self.tree)))
            transitions = [self.tree.data[i] for i in indices]
            weights = np.ones(len(indices), dtype=np.float32)
        else:
            indices, probs = self.tree.sample(batch_size)
            transitions = [self.tree.data[i] for i in indices]

            n = len(self.tree)
            weights = (1.0 / (n * probs)) ** self.beta
            weights /= weights.max()

        self.beta = np.interp(self.step_count, [0, self.beta_steps],
                              [self.beta_start, self.beta_end])
        self.step_count += 1

        return transitions, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            leaf_idx = idx + self.tree.capacity - 1
            self.tree.update(leaf_idx, priority)

    def __len__(self):
        return len(self.tree)
