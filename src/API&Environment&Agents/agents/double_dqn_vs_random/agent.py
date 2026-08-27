import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.double_dqn_vs_random.network import Network
from agents.double_dqn_vs_random.replay_buffer import ReplayBuffer


class Agent:

    def __init__(self, num_planes, rows, columns, num_actions, device="cpu",
                 lr=5e-5, gamma=0.0, buffer_capacity=100_000,
                 batch_size=512, replay_warmup=5_000, target_sync_every=10_000):

        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.replay_warmup = replay_warmup

        raw_policy_net = Network(num_planes, rows, columns, num_actions)
        raw_target_net = Network(num_planes, rows, columns, num_actions)

        self.policy_net = raw_policy_net.to(self.device)
        self.target_net = raw_target_net.to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.train_steps = 0

        self.last_mean_legal_q = 0.0
        self.last_max_legal_q = 0.0
        self.last_min_legal_q = 0.0

        self.last_mean_td_target = 0.0
        self.last_max_td_target = 0.0
        self.last_min_td_target = 0.0

        self.last_mean_td_abs = 0.0
        self.last_max_td_abs = 0.0

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float) -> int:

        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")

        if random.random() < epsilon:
            return int(np.random.choice(legal_indices))

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

            masked_q = np.full(self.num_actions, -np.inf, dtype=np.float32)
            masked_q[legal_indices] = q_values[legal_indices]

            return int(np.argmax(masked_q))

    def store(self, state, action, reward_white, next_state, done, next_legal_mask):

        self.replay_buffer.push(state, action, reward_white, next_state, done, next_legal_mask)

    def set_legal_q_stats(self, state, legal_mask) -> None:
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            return None

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)

            indices = torch.tensor(legal_indices, dtype=torch.long, device=self.device)
            legal_q = q_values[indices]

            mean_q, max_q, min_q = (
                legal_q.mean().item(),
                legal_q.max().item(),
                legal_q.min().item()
            )

            self.last_mean_legal_q = mean_q
            self.last_max_legal_q = max_q
            self.last_min_legal_q = min_q

        return None

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.replay_warmup:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward_white for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.from_numpy(np.stack([t.next_legal_mask for t in batch])).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_value = torch.zeros(
                self.batch_size,
                dtype=torch.float32,
                device=self.device
            )

            non_terminal = ~dones.bool()

            if non_terminal.any():
                next_states_nt = next_states[non_terminal]
                next_masks_nt = next_masks[non_terminal]

                next_q_policy = self.policy_net(next_states_nt)
                next_q_target = self.target_net(next_states_nt)

                next_q_policy = next_q_policy.masked_fill(
                    ~next_masks_nt.bool(),
                    -float("inf")
                )

                next_actions = next_q_policy.argmax(dim=1)

                next_q_value[non_terminal] = next_q_target.gather(1,next_actions.unsqueeze(1)).squeeze(1)

            target = rewards + self.gamma * next_q_value

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=5.0
        )

        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss = loss.item()
        print(loss)
        return loss

    def calculate_td_loss(self, state, action, reward,
                          next_state, done, next_legal_mask,
                          collect_diagnostics=False) -> float:

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_state_t = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        done_t = torch.tensor([done], dtype=torch.float32, device=self.device)
        next_mask_t = torch.from_numpy(next_legal_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.policy_net(state_t).gather(
                1, action_t.unsqueeze(1)
            ).squeeze(1)

            next_q = torch.zeros(
                1,
                dtype=torch.float32,
                device=self.device
            )

            non_terminal = ~done_t.bool()

            if non_terminal.any():
                next_policy = self.policy_net(next_state_t)
                next_target = self.target_net(next_state_t)

                next_policy = next_policy.masked_fill(
                    ~next_mask_t.bool(),
                    -float("inf")
                )

                next_action = next_policy.argmax(dim=1)

                next_q = next_target.gather(
                    1,
                    next_action.unsqueeze(1)
                ).squeeze(1)

            target = reward_t + self.gamma * next_q

        loss = self.loss_fn(q, target)

        if collect_diagnostics:
            td_error = target - q

            self.last_mean_td_target = target.mean().item()
            self.last_max_td_target = target.max().item()
            self.last_min_td_target = target.min().item()

            self.last_mean_td_abs = td_error.abs().mean().item()
            self.last_max_td_abs = td_error.abs().max().item()

        return loss.item()
