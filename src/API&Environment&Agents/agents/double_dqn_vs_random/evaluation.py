from collections import Counter
import numpy as np
from domain.configs import MAX_STEPS_PER_EPISODE, LOG_EVERY_EPISODE, EVALUATE_GAMES
from environment.grenight_environment import GrenightEnvironment
from agents.double_dqn_vs_random.agent import Agent


def evaluate_agent(env: GrenightEnvironment, agent: Agent) -> None:
    current_agent_step = 0
    log_q_every = EVALUATE_GAMES // 10

    eval_losses = []

    recent_outcomes = Counter()
    q_averages = []
    q_maxs = []
    q_mins = []

    td_target_averages = []
    td_target_maxs = []
    td_target_mins = []

    td_abs_averages = []
    td_abs_maxs = []

    for _ in range(EVALUATE_GAMES):
        state = env.reset()
        done = False
        move_count = 0

        while not done and move_count < MAX_STEPS_PER_EPISODE:

            white_old_state = state
            old_legal_mask = env.action_mask()

            white_action = agent.select_action(white_old_state, old_legal_mask, 0)
            next_white_state, white_reward, done, info = env.step(white_action)

            current_agent_step += 1
            move_count += 1

            if current_agent_step % log_q_every == 0:
                agent.set_legal_q_stats(white_old_state, old_legal_mask)

                q_averages.append(agent.last_mean_legal_q)
                q_mins.append(agent.last_min_legal_q)
                q_maxs.append(agent.last_max_legal_q)

            if not done and move_count < MAX_STEPS_PER_EPISODE:
                black_action = env.sample()
                next_white_state, black_reward, done, info = env.step(black_action)
                white_reward = -black_reward

                move_count += 1

            next_legal_mask_white = env.action_mask()

            loss = agent.calculate_td_loss(
                white_old_state,
                white_action,
                white_reward,
                next_white_state,
                done,
                next_legal_mask_white,
                current_agent_step % log_q_every == 0
            )

            eval_losses.append(loss)

            if current_agent_step % log_q_every == 0:
                td_target_averages.append(agent.last_mean_td_target)
                td_target_maxs.append(agent.last_max_td_target)
                td_target_mins.append(agent.last_min_td_target)

                td_abs_averages.append(agent.last_mean_td_abs)
                td_abs_maxs.append(agent.last_max_td_abs)

            state = next_white_state

        if not done:
            recent_outcomes["truncated"] += 1

        elif white_reward == 0.0:
            recent_outcomes["draw"] += 1

        else:
            is_winner_white = white_reward == 1
            recent_outcomes["agent_win" if is_winner_white else "random_win"] += 1

    process_stats(recent_outcomes, eval_losses, q_averages, q_maxs, q_mins, False,
                  td_target_averages, td_target_maxs, td_target_mins, td_abs_averages, td_abs_maxs)


def process_stats(outcomes: Counter,
                  losses: list[float],
                  q_averages: list[float],
                  q_maxs: list[float],
                  q_mins: list[float],
                  is_training_stats: bool,
                  td_target_averages: list[float] | None=None,
                  td_target_maxs: list[float] | None=None,
                  td_target_mins: list[float] | None=None,
                  td_abs_averages: list[float]| None=None,
                  td_abs_maxs: list[float] | None=None) -> None:


    if is_training_stats:
        avg_loss = np.mean(losses[-5000:]) if losses else float("nan")
    else:
        avg_loss = np.mean(losses) if losses else float("nan")

    completed = (
            outcomes.get("agent_win", 0)
            + outcomes.get("random_win", 0)
            + outcomes.get("draw", 0)
    )

    total_episodes = completed + outcomes.get("truncated", 0)

    agent_win_pct = (
        100.0 * outcomes.get("agent_win", 0) / completed
        if completed > 0 else 0.0
    )

    random_win_pct = (
        100.0 * outcomes.get("random_win", 0) / completed
        if completed > 0 else 0.0
    )

    draw_pct = (
        100.0 * outcomes.get("draw", 0) / completed
        if completed > 0 else 0.0
    )

    truncated_pct = (
        100.0 * outcomes.get("truncated", 0) / total_episodes
        if total_episodes > 0 else 0.0
    )

    n = LOG_EVERY_EPISODE if is_training_stats else EVALUATE_GAMES
    label = "training" if is_training_stats else "evaluation"

    print()

    if is_training_stats:
        print(f"{label} statistics — last {n:,} episodes")
    else:
        print(f"{label} statistics — another new {n:,} games")


    print(
        f"  outcomes    "
        f"agent {agent_win_pct:5.1f}%   "
        f"random {random_win_pct:5.1f}%   "
        f"draw {draw_pct:5.1f}%   "
        f"truncated {truncated_pct:5.1f}%"
    )

    print(f"  distribution {dict(outcomes)}")

    print(
        f"  agent       "
        f"loss {avg_loss:10.8f}   "
        f"Q avg {np.mean(q_averages):8.4f}   "
        f"Q max {np.mean(q_maxs):8.4f}   "
        f"Q min {np.mean(q_mins):8.4f}"
    )

    if not is_training_stats:
        print(
            f"  diagnostics "
            f"target avg {np.mean(td_target_averages):8.4f}   "
            f"target max {np.mean(td_target_maxs):8.4f}   "
            f"target min {np.mean(td_target_mins):8.4f}"
        )

        print(
            f"              "
            f"|TD| avg {np.mean(td_abs_averages):8.4f}   "
            f"|TD| max {np.mean(td_abs_maxs):8.4f}"
        )
