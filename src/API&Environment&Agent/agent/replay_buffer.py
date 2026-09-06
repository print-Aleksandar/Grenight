import random


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


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self._pos = 0

    def push(self, *args) -> None:
        transition = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
