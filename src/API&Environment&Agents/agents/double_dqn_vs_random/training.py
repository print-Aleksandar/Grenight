import os
import numpy as np
import torch
from collections import Counter
from environment.grenight_environment import GrenightEnvironment
from agents.double_dqn_vs_random.agent import Agent
from domain.configs import (
    MAX_STEPS_PER_EPISODE,
    TRAIN_EPISODES,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_EPISODES,
    CHECKPOINT_EVERY_EPISODES,
    LOG_EVERY,
    CHECKPOINT_DIR
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"will save checkpoints in: {CHECKPOINT_DIR}")

env = GrenightEnvironment()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

agent = Agent(
    num_planes=env.state_encoder.NUM_PLANES,
    rows=5,
    columns=4,
    num_actions=env.action_encoder.NUM_ACTIONS,
    device=device
)

print(f"policy_net device: {next(agent.policy_net.parameters()).device}")

"""
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "epXXXXX.pt"
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
agent.target_net.load_state_dict(checkpoint["target_state_dict"])
agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
"""


def epsilon_at_linear(ep: int) -> float:
    if ep >= EPSILON_DECAY_EPISODES:
        return EPSILON_END

    return EPSILON_START + ((EPSILON_END - EPSILON_START) * (ep / EPSILON_DECAY_EPISODES))


def save_checkpoint(ep: int):
    path = os.path.join(CHECKPOINT_DIR, f"ep{ep}.pt")
    torch.save({
        "episode": ep,
        "policy_state_dict": agent.policy_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict()
    }, path)
    print(f"[checkpoint] saved: {path}")


losses = []
recent_outcomes = Counter()

try:
    for episode in range(1, TRAIN_EPISODES + 1):

        state = env.reset()
        done = False
        move_count = 0

        while not done and move_count < MAX_STEPS_PER_EPISODE:

            white_old_state = state
            legal_mask = env.action_mask()
            epsilon = epsilon_at_linear(episode)

            white_action = agent.select_action(white_old_state, legal_mask, epsilon)
            next_state, white_reward, done, info = env.step(white_action)
            move_count += 1

            if not done and move_count < MAX_STEPS_PER_EPISODE:
                black_action = env.sample()
                next_state, black_reward, done, info = env.step(black_action)
                white_reward = -1 if black_reward == 1 else black_reward
                move_count += 1

            if not done:
                next_legal_mask_white = env.action_mask()
            else:
                next_legal_mask_white = legal_mask

            agent.store(white_old_state, white_action, white_reward, next_state, done, next_legal_mask_white)

            state = next_state

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        if not done:
            recent_outcomes["truncated"] += 1

        elif white_reward in (-0.2, -0.5):
            recent_outcomes["draw"] += 1

        else:
            is_winner_white = white_reward == 1
            recent_outcomes["agent_win" if is_winner_white else "random_win"] += 1

        if episode % CHECKPOINT_EVERY_EPISODES == 0:
            save_checkpoint(episode)

        if episode % LOG_EVERY == 0:
            avg_loss = np.mean(losses[-5000:]) if losses else float("nan")

            was_last_game_truncated = False
            was_last_game_draw = False
            was_last_game_win_for_white = False

            if not done:
                was_last_game_truncated = True

            elif white_reward in (-0.2, -0.5):
                was_last_game_draw = True

            else:
                is_winner_white = white_reward == 1
                was_last_game_win_for_white = is_winner_white

            agent.set_legal_q_stats(white_old_state, legal_mask)

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
                f"[episode {episode} logs]: "
                f"last_agent_reward: {white_reward} "
                f"was_last_game_truncated: {was_last_game_truncated} "
                f"was_last_game_draw: {was_last_game_draw} "
                f"was_last_game_win_for_white: {was_last_game_win_for_white} "
                f"avg_legal_q_of_last_white_move={agent.last_mean_legal_q:>8.4f} "
                f"max_legal_q_of_last_white_move={agent.last_max_legal_q:>8.4f} "
                f"min_legal_q_of_last_white_move={agent.last_min_legal_q:>8.4f} "
                f"ep={episode:>7} "
                f"eps={epsilon:.3f} "
                f"avg_loss={avg_loss:>8.4f} "
                f"agent_win={agent_win_pct:5.1f}% "
                f"random_win={random_win_pct:5.1f}% "
                f"draw={draw_pct:5.1f}% "
                f"truncated={truncated_pct:5.1f}% "
                f"outcomes(last{LOG_EVERY})={dict(recent_outcomes)}\n"
            )

            recent_outcomes.clear()

except KeyboardInterrupt:
    print("\n[interrupted] saving checkpoint before exit...")
    save_checkpoint(episode)

finally:
    save_checkpoint(episode)
    print("Done.")
