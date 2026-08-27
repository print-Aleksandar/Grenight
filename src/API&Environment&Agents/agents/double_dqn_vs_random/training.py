import os
import torch
from collections import Counter

from agents.double_dqn_vs_random.evaluation import process_stats, evaluate_agent
from environment.grenight_environment import GrenightEnvironment
from agents.double_dqn_vs_random.agent import Agent
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

agent = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

print(f"policy_net device: {next(agent.policy_net.parameters()).device}")

agent_step = 0
global_step = 0
episode_start = 1

"""
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "epXXXXX.pt"
)

"""
checkpoint = torch.load("checkpoints/ep3019.pt", map_location=device, weights_only=False)

episode_start = checkpoint["episode"] + 1
agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent.target_net.load_state_dict(checkpoint["target_state_dict"])
agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent.replay_buffer = checkpoint["replay_buffer"]
agent.train_steps = checkpoint["train_steps"]
agent_step = checkpoint["agent_step"]
global_step = checkpoint["global_step"]


def epsilon_at_linear(step: int) -> float:

    return max(EPSILON_END,
               EPSILON_START - (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY_STEPS)


def save_checkpoint(ep: int):
    path = os.path.join(CHECKPOINT_DIR, f"ep{ep}.pt")
    torch.save({
        "episode": ep,
        "policy_state_dict": agent.policy_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "replay_buffer": agent.replay_buffer,
        "train_steps": agent.train_steps,
        "agent_step": agent_step,
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

            white_old_state = state
            old_legal_mask = env.action_mask()
            epsilon = epsilon_at_linear(agent_step)

            white_action = agent.select_action(white_old_state, old_legal_mask, epsilon)
            next_white_state, white_reward, done, info = env.step(white_action)

            agent_step += 1
            global_step += 1
            move_count += 1

            if agent_step % LOG_Q_EVERY_STEPS == 0:
                agent.set_legal_q_stats(white_old_state, old_legal_mask)

                q_averages.append(agent.last_mean_legal_q)
                q_mins.append(agent.last_min_legal_q)
                q_maxs.append(agent.last_max_legal_q)

            if not done and move_count < MAX_STEPS_PER_EPISODE:
                black_action = env.sample()
                next_white_state, black_reward, done, info = env.step(black_action)
                white_reward = -black_reward

                global_step += 1
                move_count += 1

            next_legal_mask_white = env.action_mask()

            agent.store(white_old_state, white_action, white_reward, next_white_state, done, next_legal_mask_white)

            state = next_white_state

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        if not done:
            recent_outcomes["truncated"] += 1

        elif white_reward == 0.0:
            recent_outcomes["draw"] += 1

        else:
            is_winner_white = white_reward == 1
            recent_outcomes["agent_win" if is_winner_white else "random_win"] += 1

        if episode % CHECKPOINT_EVERY_EPISODES == 0:
            save_checkpoint(episode)

        if episode % LOG_EVERY_EPISODE == 0:
            print(f"[episode {episode} logs]: "
                  f"epsilon: {epsilon:>8.4f} "
                  f"agent_step {agent_step:>8} "
                  f"global_step {global_step:>8} "
            )

            print(f"last loss: {losses[-1]:>8.4f} ")

            process_stats(recent_outcomes, losses, q_averages, q_maxs, q_mins, True)

            evaluate_agent(env, agent)

            print(f"[episode {episode} logs]: end")

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
