import random
import numpy as np
import torch
from torch import optim, nn
from domain.configs import (
    LEARNING_RATE,
    DISCOUNT_FACTOR_GAMMA,
    BUFFER_CAPACITY,
    REPLAY_WARMUP,
    BATCH_SIZE,
    BULK_UPDATE_EVERY_TRAIN_STEPS,
    SOFT_UPDATE_TAU,
)
from agent.network import Network
from agent.replay_buffer import ReplayBuffer


class GrenightAgent:

    def __init__(self, is_self_play: bool,
                 is_double_net: bool,
                 is_dueling_net: bool,
                 is_residual_net: bool,
                 num_planes: int,
                 rows: int,
                 columns: int,
                 num_actions: int,
                 device: str|None="cpu",
                 is_bulk_update: bool|None=True) -> None:

        self.is_self_play = is_self_play
        self.is_double_net = is_double_net
        self.is_dueling_net = is_dueling_net
        self.is_residual_net = is_residual_net
        self.is_bulk_update = is_bulk_update

        self.device = torch.device(device)
        self.num_actions = num_actions

        self.gamma = DISCOUNT_FACTOR_GAMMA
        self.batch_size = BATCH_SIZE
        self.replay_warmup = REPLAY_WARMUP
        self.sync_every_steps = BULK_UPDATE_EVERY_TRAIN_STEPS
        self.tau = SOFT_UPDATE_TAU

        raw_policy_net = Network(self.is_dueling_net, self.is_residual_net, num_planes, rows, columns, num_actions)
        self.policy_net = raw_policy_net.to(self.device)

        if self.is_double_net:
            raw_target_net = Network(self.is_dueling_net, self.is_residual_net, num_planes, rows, columns, num_actions)
            self.target_net = raw_target_net.to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net = None

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

        self.train_steps = 0

        self.last_mean_legal_q = 0.0
        self.last_max_legal_q = 0.0
        self.last_min_legal_q = 0.0

        self.last_td_target = 0.0
        self.last_td_abs = 0.0

    def _q(self, net, state_t, legal_mask_t=None):
        if self.is_dueling_net:
            return net(state_t, legal_mask_t)
        return net(state_t)

    def _next_q(self, next_states_t, next_masks_t):
        next_q_policy = self._q(self.policy_net, next_states_t, next_masks_t)
        next_q_policy = next_q_policy.masked_fill(
            ~next_masks_t.bool(),
            -float("inf")
        )

        if self.is_double_net:
            next_actions = next_q_policy.argmax(dim=1)
            next_q_target = self._q(self.target_net, next_states_t, next_masks_t)
            return next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            return next_q_policy.max(dim=1).values

    def select_action(self, state: np.ndarray,
                      legal_mask: np.ndarray,
                      epsilon: float) -> int:

        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")

        if random.random() < epsilon:
            return int(np.random.choice(legal_indices))

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            legal_mask_t = (
                None if not self.is_dueling_net
                else torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
            )
            q_values = self._q(self.policy_net, state_t, legal_mask_t).squeeze(0).cpu().numpy()

            masked_q = np.full(self.num_actions, -np.inf, dtype=np.float32)
            masked_q[legal_indices] = q_values[legal_indices]

            return int(np.argmax(masked_q))

    def store(self, *args) -> None:
        self.replay_buffer.push(*args)

    def update_registry(self) -> None:
        if not self.is_double_net:
            return None

        return self.bulk_update() if self.is_bulk_update else self.soft_update()

    def bulk_update(self) -> None:
        if not self.is_double_net:
            return None

        if self.train_steps % self.sync_every_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return None

    def soft_update(self) -> None:
        if not self.is_double_net:
            return None

        with torch.no_grad():
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return None

    def set_legal_q_stats(self, state, legal_mask) -> None:
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            return None

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            legal_mask_t = (
                None if not self.is_dueling_net
                else torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
            )

            q_values = self._q(self.policy_net, state_t, legal_mask_t).squeeze(0)

            indices = torch.tensor(
                legal_indices,
                dtype=torch.long,
                device=self.device
            )

            legal_q = q_values[indices]

            mean_q = legal_q.mean().item()
            max_q = legal_q.max().item()
            min_q = legal_q.min().item()

            self.last_mean_legal_q = mean_q
            self.last_max_legal_q = max_q
            self.last_min_legal_q = min_q

        return None

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.replay_warmup:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device)
        masks = (
            None if not self.is_dueling_net
            else torch.from_numpy(np.stack([t.legal_mask for t in batch])).to(self.device)
        )
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.from_numpy(np.stack([t.next_legal_mask for t in batch])).to(self.device)

        q_values = self._q(self.policy_net, states, masks).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_value = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

            non_terminal = ~dones.bool()

            if non_terminal.any():
                next_states_nt = next_states[non_terminal]
                next_masks_nt = next_masks[non_terminal]

                next_q_value[non_terminal] = self._next_q(next_states_nt, next_masks_nt)

            if self.is_self_play:
                target = rewards - self.gamma * next_q_value
            else:
                target = rewards + (self.gamma * self.gamma) * next_q_value

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.train_steps += 1

        self.update_registry()

        return loss.item()

    def calculate_td_loss(self, state, legal_mask, action, reward,
                          next_state, done, next_legal_mask,
                          collect_diagnostics=False) -> float:

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        legal_mask_t = (
            None if not self.is_dueling_net
            else torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
        )
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_state_t = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        done_t = torch.tensor([done], dtype=torch.float32, device=self.device)
        next_mask_t = torch.from_numpy(next_legal_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self._q(self.policy_net, state_t, legal_mask_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

            next_q = torch.zeros(1, dtype=torch.float32, device=self.device)

            non_terminal = ~done_t.bool()

            if non_terminal.any():
                next_q = self._next_q(next_state_t, next_mask_t)

            if self.is_self_play:
                target = reward_t - self.gamma * next_q
            else:
                target = reward_t + (self.gamma * self.gamma) * next_q

        loss = self.loss_fn(q, target)

        if collect_diagnostics:
            td_error = target - q

            self.last_td_target = target.mean().item()
            self.last_td_abs = td_error.abs().mean().item()

        return loss.item()
