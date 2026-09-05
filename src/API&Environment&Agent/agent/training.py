import os
from collections import Counter
import torch
from agent.evaluation import process_stats, evaluate_agent_by_all_combos
from domain.configs import (
    ROWS,
    COLUMNS,
    MAX_STEPS_PER_EPISODE,
    TRAIN_EPISODES,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    CHECKPOINT_EVERY_EPISODES,
    LOG_EVERY_EPISODE,
    LOG_Q_EVERY_STEPS,
    DISCOUNT_FACTOR_GAMMA,
    CHECKPOINT_DIR_KAGGLE as CHECKPOINT_DIR
)
from agent.grenight_agent import GrenightAgent
from environment.grenight_environment import GrenightEnvironment


device = "cuda" if torch.cuda.is_available() else "cpu"


def epsilon_at(step: int) -> float:

    return max(EPSILON_END,
               EPSILON_START - (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY_STEPS)


def save_checkpoint(agent: GrenightAgent,
                    ep: int, agent_step: int,
                    is_double_net: bool) -> None:
    path = os.path.join(CHECKPOINT_DIR, f"current_implementation_ep{ep}.pt")

    if is_double_net:
        torch.save({
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "train_steps": agent.train_steps,
            "episode": ep,
            "agent_step": agent_step,
        }, path)

    else:
        torch.save({
            "policy_state_dict": agent.policy_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "train_steps": agent.train_steps,
            "episode": ep,
            "agent_step": agent_step,
        }, path)

    print(f"[checkpoint] saved: {path}")


def load_checkpoint(agent: GrenightAgent,
                    is_double_net: bool) -> tuple[GrenightAgent, int, int]:

    checkpoint = torch.load(
     "path.checkpoint.pt",
        map_location=device,
        weights_only=False
    )

    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    if is_double_net:
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.train_steps = checkpoint["train_steps"]

    episode = checkpoint["episode"]
    agent_step = checkpoint["agent_step"]

    return agent, episode, agent_step


def train_self_play_episode(env: GrenightEnvironment, agent: GrenightAgent,
                            agent_step: int, losses: list[float], q_averages: list[float],
                            q_maxs: list[float], q_mins: list[float]) -> tuple[bool, bool, bool, int]:

    state = env.reset()
    done = False
    is_draw = False
    is_white_on_turn = True
    move_count = 0

    while not done and move_count < MAX_STEPS_PER_EPISODE:
        is_white_on_turn = env.is_white_on_turn

        epsilon = epsilon_at(agent_step)
        legal_mask = env.action_mask()

        action = agent.select_action(state, legal_mask, epsilon)
        new_state, reward, done, is_draw, _ = env.step(action)
        agent_step += 1

        next_legal_mask = env.action_mask()
        agent.store(state, legal_mask, action, reward, new_state, done, next_legal_mask)

        loss = None
        if agent_step % 4 == 0:
            loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        if agent_step % LOG_Q_EVERY_STEPS == 0:
            agent.set_legal_q_stats(state, legal_mask)

            q_averages.append(agent.last_mean_legal_q)
            q_mins.append(agent.last_min_legal_q)
            q_maxs.append(agent.last_max_legal_q)

        state = new_state
        move_count += 1

    return done, is_draw, is_white_on_turn, agent_step


def train_vs_random_episode(env: GrenightEnvironment, agent: GrenightAgent,
                            agent_step: int, losses: list[float], q_averages: list[float],
                            q_maxs: list[float], q_mins: list[float]) -> tuple[bool, bool, bool, int]:

    state = env.reset()
    done = False
    is_draw = False
    is_white_on_turn = True
    move_count = 0

    while not done and move_count < MAX_STEPS_PER_EPISODE:
        is_white_on_turn = True
        white_old_state = state
        old_legal_mask = env.action_mask()
        epsilon = epsilon_at(agent_step)

        white_action = agent.select_action(white_old_state, old_legal_mask, epsilon)
        next_white_state, white_reward, done, is_draw, info = env.step(white_action)

        agent_step += 1
        move_count += 1

        total_reward = white_reward

        if not done and move_count < MAX_STEPS_PER_EPISODE:
            is_white_on_turn = False
            black_action = env.sample()
            next_white_state, black_reward, done, is_draw, info = env.step(
                black_action
            )

            move_count += 1

            total_reward -= DISCOUNT_FACTOR_GAMMA * black_reward

        next_legal_mask = env.action_mask()

        agent.store(
            white_old_state,
            old_legal_mask,
            white_action,
            total_reward,
            next_white_state,
            done,
            next_legal_mask,
        )

        if agent_step % LOG_Q_EVERY_STEPS == 0:
            agent.set_legal_q_stats(white_old_state, old_legal_mask)

            q_averages.append(agent.last_mean_legal_q)
            q_mins.append(agent.last_min_legal_q)
            q_maxs.append(agent.last_max_legal_q)

        if agent_step % 2 == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        state = next_white_state

    return done, is_draw, is_white_on_turn, agent_step


def train_agent(is_self_play: bool,
                is_double_net: bool,
                is_dueling_net: bool,
                is_residual_net: bool,
                is_canonical_version: bool,
                will_store_history_in_state: bool,
                will_do_reward_shaping: bool,
                is_bulk_update: bool | None=True) -> None:

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"will save checkpoints in: {CHECKPOINT_DIR}")

    env = GrenightEnvironment(
        is_canonical_version=is_canonical_version,
        will_store_history_in_state=will_store_history_in_state,
        will_do_reward_shaping=will_do_reward_shaping
    )

    agent = GrenightAgent(
        is_self_play=is_self_play,
        is_double_net=is_double_net,
        is_dueling_net=is_dueling_net,
        is_residual_net=is_residual_net,
        rows=ROWS,
        columns=COLUMNS,
        num_actions=env.action_encoder.num_actions,
        num_planes=env.state_encoder.num_planes,
        device=device,
        is_bulk_update=is_bulk_update
    )

    print(f"policy_net device: {next(agent.policy_net.parameters()).device}")

    agent_step = 0
    episode_start = 1

    losses = []
    recent_outcomes = Counter()
    q_averages = []
    q_maxs = []
    q_mins = []

    try:
        for episode in range(episode_start, TRAIN_EPISODES + 1):

            if is_self_play:
                done, is_draw, is_white_on_turn, agent_step = train_self_play_episode(
                    env, agent, agent_step, losses, q_averages, q_maxs, q_mins
                )
            else:
                done, is_draw, is_white_on_turn, agent_step = train_vs_random_episode(
                    env, agent, agent_step, losses, q_averages, q_maxs, q_mins
                )

            if not done:
                recent_outcomes["truncated"] += 1

            else:
                if is_draw:
                    recent_outcomes["draw"] += 1
                else:
                    recent_outcomes["white_win" if is_white_on_turn else "black_win"] += 1

            if episode % CHECKPOINT_EVERY_EPISODES == 0:
                save_checkpoint(agent, episode, agent_step)

            if episode % LOG_EVERY_EPISODE == 0:
                print()
                print("─" * 72)
                print(f"  EPISODE {episode:,}")
                print()

                print(
                    f"  ε (epsilon)     : {epsilon_at(agent_step):>8.4f}\n"
                    f"  agent step      : {agent_step:>8,}"
                )

                print()

                process_stats(recent_outcomes, losses, q_averages, q_maxs, q_mins, True, True, is_self_play)

                print()

                print("  Running evaluation...")
                evaluate_agent_by_all_combos(env, agent, is_self_play)

                print("─" * 72)

                print()

                recent_outcomes.clear()
                q_averages = []
                q_maxs = []
                q_mins = []


    except KeyboardInterrupt:
        print("\n[interrupted] saving checkpoint before exit...")
        save_checkpoint(agent, episode, agent_step, is_double_net)

    finally:
        save_checkpoint(agent, episode, agent_step, is_double_net)
        print("Done.")

train_agent(True, False, False, False, False, False, False)
