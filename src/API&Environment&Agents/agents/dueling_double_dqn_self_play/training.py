import os
import random

import torch
from collections import Counter
from agents.dueling_double_dqn_self_play.evaluation import process_stats, evaluate_agent_by_all_combos
from environment.canonical_version.grenight_environment import GrenightEnvironment
from agents.dueling_double_dqn_self_play.agent import Agent
from domain.configs import (
    MAX_STEPS_PER_EPISODE,
    TRAIN_EPISODES,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    CHECKPOINT_EVERY_EPISODES,
    LOG_EVERY_EPISODE,
    CHECKPOINT_DIR, LOG_Q_EVERY_STEPS
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"will save checkpoints in: {CHECKPOINT_DIR}")

env = GrenightEnvironment()

device = "cuda" if torch.cuda.is_available() else "cpu"

current_agent = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load("/kaggle/input/datasets/mojavoda/grenight-ddqn-self-play/ep10000.pt", map_location=device, weights_only=False)
checkpoint.policy_net.load_state_dict(checkpoint["policy_state_dict"])
checkpoint.target_net.load_state_dict(checkpoint["target_state_dict"])
checkpoint.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
checkpoint.replay_buffer = checkpoint["replay_buffer"]
checkpoint.train_steps = checkpoint["train_steps"]

agent_5k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load("/kaggle/input/datasets/mojavoda/grenight-ddqn-self-play/ep5000.pt", map_location=device, weights_only=False)
checkpoint.policy_net.load_state_dict(checkpoint["policy_state_dict"])
checkpoint.target_net.load_state_dict(checkpoint["target_state_dict"])
checkpoint.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
checkpoint.replay_buffer = checkpoint["replay_buffer"]
checkpoint.train_steps = checkpoint["train_steps"]

print(f"policy_net device: {next(current_agent.policy_net.parameters()).device}")

global_step = 0
episode_start = 1


def epsilon_at(step: int) -> float:

    return max(EPSILON_END,
               EPSILON_START - (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY_STEPS)


def save_checkpoint(ep: int):
    path = os.path.join(CHECKPOINT_DIR, f"ep{ep}.pt")
    torch.save({
        "episode": ep,
        "policy_state_dict": current_agent.policy_net.state_dict(),
        "target_state_dict": current_agent.target_net.state_dict(),
        "optimizer_state_dict": current_agent.optimizer.state_dict(),
        "replay_buffer": current_agent.replay_buffer,
        "train_steps": current_agent.train_steps,
        "global_step": global_step,
    }, path)
    print(f"[checkpoint] saved: {path}")


losses = []
recent_outcomes = Counter()
q_averages = []
q_maxs = []
q_mins = []

try:
    for episode in range(episode_start, TRAIN_EPISODES + 1):

        state = env.reset()
        done = False
        move_count = 0

        while not done and move_count < MAX_STEPS_PER_EPISODE:
            epsilon = epsilon_at(global_step)
            legal_mask = env.action_mask()

            who_is_on_turn = env.is_white_on_turn

            if random.random() < 0.34:
                action = agent_5k.select_action(state, legal_mask, 0.0)
            else:
                action = current_agent.select_action(state, legal_mask, epsilon)

            new_state, reward, done, _ = env.step(action)

            if not who_is_on_turn and reward == -1.0:
                reward = -reward

            next_legal_mask = env.action_mask()

            current_agent.store(state, legal_mask, action, reward, new_state, done, next_legal_mask)
            global_step += 1

            loss = None
            if global_step % 4 == 0:
                loss = current_agent.train_step()
            if loss is not None:
                losses.append(loss)

            if global_step % LOG_Q_EVERY_STEPS == 0:
                current_agent.set_legal_q_stats(state, legal_mask)

                q_averages.append(current_agent.last_mean_legal_q)
                q_mins.append(current_agent.last_min_legal_q)
                q_maxs.append(current_agent.last_max_legal_q)

            state = new_state
            move_count += 1

        if not done:
            recent_outcomes["truncated"] += 1

        elif reward == 0.0:
            recent_outcomes["draw"] += 1

        else:
            if who_is_on_turn:
                recent_outcomes["white_win"] += 1
            else:
                recent_outcomes["black_win"] += 1

        if episode % CHECKPOINT_EVERY_EPISODES == 0:
            save_checkpoint(episode)

        if episode % LOG_EVERY_EPISODE == 0:
            print()
            print("─" * 72)
            print(f"  EPISODE {episode:,}")
            print()

            print(
                f"  ε (epsilon)     : {epsilon:>8.4f}\n"
                f"  global steps    : {global_step:>8,}"
            )

            print()

            process_stats(recent_outcomes, losses, q_averages, q_maxs, q_mins,True, True, True)

            print()

            print("  Running evaluation...")
            evaluate_agent_by_all_combos(env, current_agent)

            print("─" * 72)

            print()

            recent_outcomes.clear()
            q_averages = []
            q_maxs = []
            q_mins = []


except KeyboardInterrupt:
    print("\n[interrupted] saving checkpoint before exit...")
    save_checkpoint(episode)

finally:
    save_checkpoint(episode)
    print("Done.")
