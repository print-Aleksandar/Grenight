import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.q_network import QNetwork
from agents.replay_buffer import ReplayBuffer


class DoubleDQNAgent:

    def __init__(self, num_planes, rows, columns, num_actions, device="cpu",
                 lr=1e-4, gamma=0.99, buffer_capacity=100_000,
                 batch_size=256, target_sync_every=1000):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every

        self.policy_net = QNetwork(num_planes, rows, columns, num_actions).to(self.device)
        self.target_net = QNetwork(num_planes, rows, columns, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray,
                       is_white_turn: bool, epsilon: float) -> int:

        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")

        if random.random() < epsilon:
            return int(np.random.choice(legal_indices))

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        masked_q = np.full(
            self.num_actions,
            -np.inf if is_white_turn else np.inf,
            dtype=np.float32,
        )
        masked_q[legal_indices] = q_values[legal_indices]

        return int(np.argmax(masked_q)) if is_white_turn else int(np.argmin(masked_q))

    def store(self, state, action, reward_white, next_state, done,
              next_legal_mask, next_is_white_turn):
        self.replay_buffer.push(
            state, action, reward_white, next_state, done,
            next_legal_mask, next_is_white_turn,
        )

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward_white for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.from_numpy(np.stack([t.next_legal_mask for t in batch])).to(self.device)
        next_white_turn = torch.tensor(
            [t.next_is_white_turn for t in batch], dtype=torch.bool, device=self.device
        )

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_target = self.target_net(next_states)

            legal = next_masks.bool()

            white_q = next_q_policy.masked_fill(~legal, -float("inf"))
            white_actions = white_q.argmax(dim=1)

            black_q = next_q_target.masked_fill(~legal, 0.0)

            num_legal = legal.sum(dim=1).clamp(min=1)
            black_values = black_q.sum(dim=1) / num_legal

            white_values = next_q_target.gather(
                1, white_actions.unsqueeze(1)
            ).squeeze(1)

            next_q_value = torch.where(
                next_white_turn,
                white_values,
                black_values,
            )

            next_q_value = torch.where(
                dones.bool(),
                torch.zeros_like(next_q_value),
                next_q_value,
            )

            target = rewards + (1.0 - dones) * self.gamma * next_q_value

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
