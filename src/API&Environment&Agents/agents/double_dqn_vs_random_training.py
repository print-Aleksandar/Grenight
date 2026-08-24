import os
import numpy as np
import torch
from collections import Counter
from environment.grenight_environment import GrenightEnvironment
from agents.double_dqn_vs_random_agent import DoubleDQNAgent
from domain.configs import (
    MAX_STEPS_PER_EPISODE,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    CHECKPOINT_EVERY_EPISODES,
    LOG_EVERY_EPISODES,
    CHECKPOINT_DIR
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"will save checkpoints in: {CHECKPOINT_DIR}")

env = GrenightEnvironment()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

agent = DoubleDQNAgent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

print(f"policy_net device: {next(agent.policy_net.parameters()).device}")

AGENT_IS_WHITE = True

agent_step = 0
global_step = 0

"""
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ep15000.pt"
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent.target_net.load_state_dict(checkpoint["target_state_dict"])
agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

global_step = checkpoint["global_step"]
agent_step = checkpoint["agent_step"] 
"""


def epsilon_at(step):
    frac = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def save_checkpoint(episode: int, tag: str = ""):
    path = os.path.join(CHECKPOINT_DIR, f"double_dqn_vs_random_ep{episode}{tag}.pt")
    torch.save({
        "episode": episode,
        "global_step": global_step,
        "policy_state_dict": agent.policy_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "agent_step": agent_step,
    }, path)
    print(f"[checkpoint] saved: {path}")


episode_lengths = []
losses = []
recent_outcomes = Counter()


try:
    for episode in range(1, 50_000 + 1):

        state = env.reset()
        done = False
        move_count = 0

        while not done and move_count < MAX_STEPS_PER_EPISODE:

            acting_player_is_white = env.is_white_on_turn
            legal_mask = env.action_mask()
            is_agent_turn = (acting_player_is_white == AGENT_IS_WHITE)

            if is_agent_turn:
                epsilon = epsilon_at(agent_step)

                action = agent.select_action(
                    state,
                    legal_mask,
                    acting_player_is_white,
                    epsilon
                )

                agent_step += 1

            else:
                action = env.sample()

            next_state, reward, done, info = env.step(action)

            reward_white = reward if acting_player_is_white else -reward

            next_legal_mask = env.action_mask() if not done else np.zeros(
                agent.num_actions, dtype=bool
            )

            agent.store(
                state, action, reward_white, next_state, done,
                next_legal_mask, env.is_white_on_turn,
            )

            state = next_state
            move_count += 1
            global_step += 1

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        episode_lengths.append(move_count)

        if not done:
            recent_outcomes["truncated"] += 1

        elif reward == 0.0:
            recent_outcomes["draw"] += 1

        else:
            winner_is_white = acting_player_is_white if reward > 0 else not acting_player_is_white
            recent_outcomes["agent_win" if winner_is_white == AGENT_IS_WHITE else "random_win"] += 1

        if episode % CHECKPOINT_EVERY_EPISODES == 0:
            save_checkpoint(episode)

        if episode % LOG_EVERY_EPISODES == 0:
            avg_len = np.mean(episode_lengths[-LOG_EVERY_EPISODES:])
            avg_loss = np.mean(losses[-5000:]) if losses else float("nan")

            completed = (
                    recent_outcomes.get("agent_win", 0)
                    + recent_outcomes.get("random_win", 0)
                    + recent_outcomes.get("draw", 0)
            )

            total_episodes = completed + recent_outcomes.get("truncated", 0)

            agent_win_pct = (
                100.0 * recent_outcomes.get("agent_win", 0) / completed
                if completed > 0 else 0.0
            )

            random_win_pct = (
                100.0 * recent_outcomes.get("random_win", 0) / completed
                if completed > 0 else 0.0
            )

            draw_pct = (
                100.0 * recent_outcomes.get("draw", 0) / completed
                if completed > 0 else 0.0
            )

            truncated_pct = (
                100.0 * recent_outcomes.get("truncated", 0) / total_episodes
                if total_episodes > 0 else 0.0
            )

            print(
                f"ep={episode:>7} "
                f"step={global_step:>8} "
                f"agent_step={agent_step:>8} "
                f"eps={epsilon:.3f} "
                f"avg_len={avg_len:>6.1f} "
                f"avg_loss={avg_loss:>8.4f} "
                f"agent_win={agent_win_pct:5.1f}% "
                f"random_win={random_win_pct:5.1f}% "
                f"draw={draw_pct:5.1f}% "
                f"truncated={truncated_pct:5.1f}% "
                f"outcomes(last{LOG_EVERY_EPISODES})={dict(recent_outcomes)}"
            )

            recent_outcomes.clear()

except KeyboardInterrupt:
    print("\n[interrupted] saving checkpoint before exit...")
    save_checkpoint(episode, tag="_interrupted")

finally:
    save_checkpoint(episode, tag="_final")
    print("Done.")
