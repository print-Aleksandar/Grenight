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

    last = LOG_EVERY_EPISODE if is_training_stats else EVALUATE_GAMES
    process = "training" if is_training_stats else "evaluation"

    print(f"\n{process} stats:")

    print(
        "outcomes:"
        f"agent_win={agent_win_pct:5.1f}% "
        f"random_win={random_win_pct:5.1f}% "
        f"draw={draw_pct:5.1f}% "
        f"truncated={truncated_pct:5.1f}% "
        f"outcomes(last{last})={dict(outcomes)}"
    )

    print(
        "agent:"
        f"avg_loss={avg_loss:>8.8f} "
        f"avg_q={np.mean(q_averages):>8.4f} "
        f"avg_max_q={np.mean(q_maxs):>8.4f} "
        f"avg_min_q={np.mean(q_mins):>8.4f}"
    )

    if not is_training_stats:
        print(
            "additional_diagnostics:"
            f"target_averages={np.mean(td_target_averages):>8.4f} "
            f"target_maxs={np.mean(td_target_maxs):>8.4f} "
            f"target_mins={np.mean(td_target_mins):>8.4f} "
            f"td_abs_averages={np.mean(td_abs_averages):>8.4f} "
            f"td_abs_maxs={np.mean(td_abs_maxs):>8.4f}\n"
        )
    else:
        print()
