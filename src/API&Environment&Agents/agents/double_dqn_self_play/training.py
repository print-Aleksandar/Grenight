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

print(f"policy_net device: {next(current_agent.policy_net.parameters()).device}")

global_step = 0
episode_start = 1

checkpoint = torch.load("/kaggle/input/datasets/mojavoda/grenight-ddqn-self-play/ep60000.pt", map_location=device, weights_only=False)
episode_start = checkpoint["episode"] + 1
current_agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
current_agent.target_net.load_state_dict(checkpoint["target_state_dict"])
current_agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
current_agent.replay_buffer = checkpoint["replay_buffer"]
current_agent.train_steps = checkpoint["train_steps"]
global_step = checkpoint["global_step"]

"""
agent_10k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load(f"/content/Grenight/src/API&Environment&Agents/agents/dueling_double_dqn_self_play/checkpoints/ep10000.pt", map_location=device, weights_only=False)
agent_10k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_10k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_10k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_10k.replay_buffer = checkpoint["replay_buffer"]
agent_10k.train_steps = checkpoint["train_steps"]

agent_15k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load(f"/content/Grenight/src/API&Environment&Agents/agents/dueling_double_dqn_self_play/checkpoints/ep15000.pt", map_location=device, weights_only=False)
agent_15k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_15k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_15k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_15k.replay_buffer = checkpoint["replay_buffer"]
agent_15k.train_steps = checkpoint["train_steps"]

agent_20k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load(f"/content/Grenight/src/API&Environment&Agents/agents/dueling_double_dqn_self_play/checkpoints/ep20000.pt", map_location=device, weights_only=False)
agent_20k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_20k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_20k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_20k.replay_buffer = checkpoint["replay_buffer"]
agent_20k.train_steps = checkpoint["train_steps"]

FOR 25_000-40_000 EPISODE TRAINING WERE USED 10, 15 AND 20K CHECKPOINTS

agent_25k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load(f"/content/Grenight/src/API&Environment&Agents/agents/dueling_double_dqn_self_play/checkpoints/ep25000.pt", map_location=device, weights_only=False)
agent_25k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_25k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_25k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_25k.replay_buffer = checkpoint["replay_buffer"]
agent_25k.train_steps = checkpoint["train_steps"]

agent_30k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load(f"/content/Grenight/src/API&Environment&Agents/agents/dueling_double_dqn_self_play/checkpoints/ep30000.pt", map_location=device, weights_only=False)
agent_30k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_30k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_30k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_30k.replay_buffer = checkpoint["replay_buffer"]
agent_30k.train_steps = checkpoint["train_steps"]

agent_35k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load(f"/content/Grenight/src/API&Environment&Agents/agents/dueling_double_dqn_self_play/checkpoints/ep35000.pt", map_location=device, weights_only=False)
agent_35k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_35k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_35k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_35k.replay_buffer = checkpoint["replay_buffer"]
agent_35k.train_steps = checkpoint["train_steps"]

FRO 40_000-60_000 EPISODE TRAINING WERE USED 25, 30 AND 35K CHECKPOINTS
"""

agent_45k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load("/kaggle/input/datasets/mojavoda/grenight-ddqn-self-play/ep45000.pt", map_location=device, weights_only=False)
agent_45k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_45k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_45k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_45k.replay_buffer = checkpoint["replay_buffer"]
agent_45k.train_steps = checkpoint["train_steps"]

agent_50k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load("/kaggle/input/datasets/mojavoda/grenight-ddqn-self-play/ep50000.pt", map_location=device, weights_only=False)
agent_50k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_50k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_50k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_50k.replay_buffer = checkpoint["replay_buffer"]
agent_50k.train_steps = checkpoint["train_steps"]

agent_55k = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

checkpoint = torch.load("/kaggle/input/datasets/mojavoda/grenight-ddqn-self-play/ep55000.pt", map_location=device, weights_only=False)
agent_55k.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent_55k.target_net.load_state_dict(checkpoint["target_state_dict"])
agent_55k.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent_55k.replay_buffer = checkpoint["replay_buffer"]
agent_55k.train_steps = checkpoint["train_steps"]


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

            if random.random() < 0.7:
                choice = random.choice([45, 50, 55])
                if choice == 45:
                    action = agent_45k.select_action(state, legal_mask, 0.0)
                elif choice == 50:
                    action = agent_50k.select_action(state, legal_mask, 0.0)
                else:
                    action = agent_55k.select_action(state, legal_mask, 0.0)
            else:
                action = current_agent.select_action(state, legal_mask, epsilon)

            new_state, reward, done, _ = env.step(action)

            if not who_is_on_turn and reward == -1.0:
                reward = -reward

            current_agent.store(state, action, reward, new_state, done, legal_mask)
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

        elif reward == -0.05 or reward == -0.33:
            recent_outcomes["draw"] += 1

        else:
            if who_is_on_turn:
                if reward == 1:
                    recent_outcomes["white_win"] += 1
                else:
                    recent_outcomes["black_win"] += 1
            else:
                if reward == 1:
                    recent_outcomes["black_win"] += 1
                else:
                    recent_outcomes["white_win"] += 1

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