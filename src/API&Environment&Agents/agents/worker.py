import numpy as np
import torch
from environment.grenight_environment import GrenightEnvironment
from domain.configs import MAX_STEPS_PER_EPISODE


def epsilon_at(step, epsilon_start, epsilon_end, epsilon_decay_steps):
    frac = min(step / epsilon_decay_steps, 1.0)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


def select_action_cpu(policy_net, state, legal_mask, is_white_turn, epsilon, num_actions):
    legal_indices = np.flatnonzero(legal_mask)

    if np.random.random() < epsilon:
        return int(np.random.choice(legal_indices))

    with torch.no_grad():
        state_t = torch.from_numpy(state).unsqueeze(0)
        q_values = policy_net(state_t).squeeze(0).numpy()

    masked_q = np.full(num_actions, -np.inf if is_white_turn else np.inf, dtype=np.float32)
    masked_q[legal_indices] = q_values[legal_indices]

    return int(np.argmax(masked_q)) if is_white_turn else int(np.argmin(masked_q))


def worker_loop(shared_policy_net, transition_queue, episode_queue, stop_event,
                 epsilon_start, epsilon_end, epsilon_decay_steps,
                 num_actions, agent_is_white=True):

    torch.set_num_threads(1)

    env = GrenightEnvironment()
    local_agent_step = 0

    while not stop_event.is_set():
        state = env.reset()
        done = False
        move_count = 0
        final_reward = None
        final_acting_white = None

        while not done and move_count < MAX_STEPS_PER_EPISODE:
            if stop_event.is_set():
                return

            acting_player_is_white = env.is_white_on_turn
            legal_mask = env.action_mask()
            is_agent_turn = (acting_player_is_white == agent_is_white)

            if is_agent_turn:
                epsilon = epsilon_at(local_agent_step, epsilon_start, epsilon_end, epsilon_decay_steps)
                action = select_action_cpu(
                    shared_policy_net, state, legal_mask,
                    acting_player_is_white, epsilon, num_actions
                )
                local_agent_step += 1
            else:
                action = env.sample()

            next_state, reward, done, info = env.step(action)
            reward_white = reward if acting_player_is_white else -reward
            next_legal_mask = env.action_mask() if not done else np.zeros(num_actions, dtype=bool)

            transition_queue.put((
                state, action, reward_white, next_state, done,
                next_legal_mask, env.is_white_on_turn
            ))

            final_reward = reward
            final_acting_white = acting_player_is_white

            state = next_state
            move_count += 1

        if not done:
            outcome = "truncated"
        elif final_reward == 0.0:
            outcome = "draw"
        else:
            winner_is_white = final_acting_white if final_reward > 0 else not final_acting_white
            outcome = "agent_win" if winner_is_white == agent_is_white else "random_win"

        episode_queue.put((move_count, outcome, local_agent_step))
