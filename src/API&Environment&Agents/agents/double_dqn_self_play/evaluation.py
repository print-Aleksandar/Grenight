from collections import Counter
import numpy as np
from domain.configs import MAX_STEPS_PER_EPISODE, LOG_EVERY_EPISODE, EVALUATE_GAMES
from environment.canonical_version.grenight_environment import GrenightEnvironment
from agents.double_dqn_self_play.agent import Agent


def evaluate_agent_by_all_combos(env: GrenightEnvironment, agent: Agent) -> None:
    evaluate_agent(env, agent, True, False)
    evaluate_agent(env, agent, False, True)
    evaluate_agent(env, agent, True, True)


def evaluate_agent(env: GrenightEnvironment,
                   agent: Agent,
                   is_agent_playing_for_white: bool,
                   is_agent_playing_for_black: bool) -> None:

    current_global_step = 0
    log_q_every = EVALUATE_GAMES // 10
    if not is_agent_playing_for_white or not is_agent_playing_for_black:
        log_q_every //= 2

    eval_losses = []

    recent_outcomes = Counter()
    draw_reasons = Counter()
    q_averages = []
    q_maxs = []
    q_mins = []

    td_target_values = []
    td_abs_values = []

    for _ in range(EVALUATE_GAMES):
        state = env.reset()
        done = False
        move_count = 0
        info = {}

        while not done and move_count < MAX_STEPS_PER_EPISODE:
            if env.is_white_on_turn:
                if is_agent_playing_for_white:
                    legal_mask = env.action_mask()
                    action = agent.select_action(state, legal_mask, 0)

                    new_state, reward, done, info = env.step(action)
                    current_global_step += 1

                    if current_global_step % log_q_every == 0:
                        agent.set_legal_q_stats(state, legal_mask)

                        q_averages.append(agent.last_mean_legal_q)
                        q_mins.append(agent.last_min_legal_q)
                        q_maxs.append(agent.last_max_legal_q)

                    next_legal_mask = env.action_mask()

                    loss = agent.calculate_td_loss(
                        state,
                        action,
                        reward,
                        new_state,
                        done,
                        next_legal_mask,
                        current_global_step % log_q_every == 0
                    )

                    eval_losses.append(loss)

                    if current_global_step % log_q_every == 0:
                        td_target_values.append(agent.last_td_target)
                        td_abs_values.append(agent.last_td_abs)
                else:
                    action = env.sample()
                    new_state, reward, done, _ = env.step(action)
            else:
                if is_agent_playing_for_black:
                    legal_mask = env.action_mask()
                    action = agent.select_action(state, legal_mask, 0)

                    new_state, reward, done, info = env.step(action)
                    current_global_step += 1

                    if current_global_step % log_q_every == 0:
                        agent.set_legal_q_stats(state, legal_mask)

                        q_averages.append(agent.last_mean_legal_q)
                        q_mins.append(agent.last_min_legal_q)
                        q_maxs.append(agent.last_max_legal_q)

                    next_legal_mask = env.action_mask()

                    loss = agent.calculate_td_loss(
                        state,
                        action,
                        -reward,
                        new_state,
                        done,
                        next_legal_mask,
                        current_global_step % log_q_every == 0
                    )

                    eval_losses.append(loss)

                    if current_global_step % log_q_every == 0:
                        td_target_values.append(agent.last_td_target)
                        td_abs_values.append(agent.last_td_abs)
                else:
                    action = env.sample()
                    new_state, reward, done, _ = env.step(action)

            move_count += 1
            state = new_state

        if not done:
            recent_outcomes["truncated"] += 1

        elif reward == 0.0:
            recent_outcomes["draw"] += 1
            draw_reasons[info.get("draw_reason")] += 1

        else:
            is_winner_white = reward == 1
            recent_outcomes["white_win" if is_winner_white else "black_win"] += 1

    process_stats(recent_outcomes, eval_losses, q_averages, q_maxs, q_mins,False,
                  is_agent_playing_for_white, is_agent_playing_for_black,
                  td_target_values, td_abs_values, draw_reasons)


def process_stats(outcomes: Counter,
                  losses: list[float],
                  q_averages: list[float],
                  q_maxs: list[float],
                  q_mins: list[float],
                  is_training_stats: bool,
                  is_agent_playing_for_white: bool,
                  is_agent_playing_for_black: bool,
                  td_target_values: list[float] | None=None,
                  td_abs_values: list[float] | None=None,
                  draw_reasons: Counter | None=None) -> None:


    if is_training_stats:
        avg_loss = np.mean(losses[-5000:]) if losses else float("nan")
    else:
        avg_loss = np.mean(losses) if losses else float("nan")

    completed = (
            outcomes.get("white_win", 0)
            + outcomes.get("black_win", 0)
            + outcomes.get("draw", 0)
    )

    total_episodes = completed + outcomes.get("truncated", 0)

    win_pct = (
        100.0 * outcomes.get("white_win", 0) / completed
        if completed > 0 else 0.0
    )

    black_pct = (
        100.0 * outcomes.get("black_win", 0) / completed
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

    if is_agent_playing_for_white and is_agent_playing_for_black:
        label += " self play"
    elif is_agent_playing_for_white:
        label += " (agent=white vs random)"
    else:
        label += " (agent=black vs random)"

    print()

    if is_training_stats:
        print(f"{label} statistics — last {n:,} episodes")
    else:
        print(f"{label} statistics — another new {n:,} games")


    print(
        f"  outcomes    "
        f"white {win_pct:5.1f}%   "
        f"black {black_pct:5.1f}%   "
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

    print(f"  draw reasons {dict(draw_reasons)}")

    if not is_training_stats:
        print(
            f"  diagnostics "
            f"target avg {np.mean(td_target_values):8.4f}   "
            f"target max {np.max(td_target_values):8.4f}   "
            f"target min {np.min(td_target_values):8.4f}"
        )

        print(
            f"              "
            f"|TD| avg {np.mean(td_abs_values):8.4f}   "
            f"|TD| max {np.max(td_abs_values):8.4f}"
        )
