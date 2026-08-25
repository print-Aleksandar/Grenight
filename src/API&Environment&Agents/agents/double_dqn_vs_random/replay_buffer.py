import random
from collections import deque


class Transition:
    __slots__ = ["state", "action", "reward_white", "next_state", "done", "next_legal_mask"]

    def __init__(self, state, action, reward_white, next_state, done, next_legal_mask):
        self.state = state
        self.action = action
        self.reward_white = reward_white
        self.next_state = next_state
        self.done = done
        self.next_legal_mask = next_legal_mask


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
