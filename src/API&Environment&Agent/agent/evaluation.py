from collections import Counter
from pathlib import Path
import numpy as np
import torch
from domain.configs import MAX_STEPS_PER_EPISODE, LOG_EVERY_EPISODE, EVALUATE_GAMES, DISCOUNT_FACTOR_GAMMA, ROWS, \
    COLUMNS
from environment.action_encoder import ActionEncoder
from environment.grenight_environment import GrenightEnvironment
from agent.grenight_agent import GrenightAgent
from environment.piece_plane_encoder import PiecePlaneEncoder


agent_tester = GrenightAgent(
    is_self_play=False,
    is_double_net=True,
    is_dueling_net=True,
    is_residual_net=True,
    is_bulk_update=False,
    rows=ROWS,
    columns=COLUMNS,
    num_actions=ActionEncoder(is_canonical_version=False).num_actions,
    num_planes=PiecePlaneEncoder.NUM_PLANES_ONLY_CURRENT,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


def load_checkpoint(agent: GrenightAgent,
                    is_double_net: bool) -> None:

    current_dir = Path(__file__).resolve().parent
    checkpoint_path = current_dir / "../implementations/ver50/p_111000/current_implementation_ep8000.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=False
    )

    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    if is_double_net:
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])

load_checkpoint(agent_tester, True)


def evaluate_agent_by_all_combos(env: GrenightEnvironment,
                                 agent: GrenightAgent,
                                 is_self_play: bool) -> None:

    evaluate_agent(env, agent, is_self_play,True, False)

    if is_self_play:
        evaluate_agent(env, agent, is_self_play,False, True)
        evaluate_again_against_test_agent(env, agent, agent_tester)


def evaluate_again_against_test_agent(env_arg: GrenightEnvironment,
                                      agent_to_test: GrenightAgent,
                                      agent_tester: GrenightAgent) -> None:

    outcomes = Counter()
    for _ in range(EVALUATE_GAMES):
        env_arg.reset()
        done = False
        is_draw = False
        is_white_on_turn = True
        move_count = 0

        while not done and move_count < MAX_STEPS_PER_EPISODE:
            is_white_on_turn = env_arg.is_white_on_turn

            if is_white_on_turn:
                action = agent_tester.select_action(env_arg.get_state(), env_arg.action_mask(), 0.05)
            else:
                action = agent_to_test.select_action(env_arg.get_state(), env_arg.action_mask(), 0.05)

            _, _, done, is_draw, _ = env_arg.step(action)

            move_count += 1

        if not done:
            outcomes["truncated"] += 1
        elif is_draw:
            outcomes["draw"] += 1
        else:
            outcomes["white_win" if is_white_on_turn else "black_win"] += 1

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

    label = "(self_play_agent=black vs fixed_res_dueling_ddqn_checkpoint=white)"

    print()

    print(f"evaluation {label} statistics — another new {EVALUATE_GAMES:,} games")

    print(
        f"  outcomes    "
        f"white {win_pct:5.1f}%   "
        f"black {black_pct:5.1f}%   "
        f"draw {draw_pct:5.1f}%   "
        f"truncated {truncated_pct:5.1f}%"
    )

    print(f"  distribution {dict(outcomes)}")


def evaluate_agent(env: GrenightEnvironment,
                   agent: GrenightAgent,
                   is_self_play: bool,
                   is_agent_playing_for_white: bool,
                   is_agent_playing_for_black: bool) -> None:

    current_agent_step = 0
    log_q_every = EVALUATE_GAMES // 10
    if not is_self_play:
        log_q_every //= 2

    eval_losses = []
    recent_outcomes = Counter()
    q_averages, q_maxs, q_mins = [], [], []
    td_target_values, td_abs_values = [], []

    for _ in range(EVALUATE_GAMES):
        env.reset()
        done = False
        is_draw = False
        is_white_on_turn = True
        move_count = 0

        while not done and move_count < MAX_STEPS_PER_EPISODE:
            is_white_on_turn = True
            white_old_state = env.get_state()
            white_legal_mask = env.action_mask()

            if is_agent_playing_for_white:
                white_action = agent.select_action(white_old_state, white_legal_mask, 0.0)
            else:
                white_action = env.sample()

            black_old_state, white_reward, done, is_draw, _ = env.step(white_action)
            move_count += 1

            if is_agent_playing_for_white:
                current_agent_step += 1
                collect = current_agent_step % log_q_every == 0
                if collect:
                    agent.set_legal_q_stats(white_old_state, white_legal_mask)
                    q_averages.append(agent.last_mean_legal_q)
                    q_mins.append(agent.last_min_legal_q)
                    q_maxs.append(agent.last_max_legal_q)

            black_legal_mask = env.action_mask()
            black_reward = 0.0

            if not done and move_count < MAX_STEPS_PER_EPISODE:
                if is_agent_playing_for_black:
                    black_action = agent.select_action(black_old_state, black_legal_mask, 0.0)
                else:
                    black_action = env.sample()

                is_white_on_turn = False
                white_new_state, black_reward, done, is_draw, _ = env.step(black_action)
                move_count += 1

                if is_self_play and is_agent_playing_for_black:
                    current_agent_step += 1
                    collect = current_agent_step % log_q_every == 0
                    if collect:
                        agent.set_legal_q_stats(black_old_state, black_legal_mask)
                        q_averages.append(agent.last_mean_legal_q)
                        q_mins.append(agent.last_min_legal_q)
                        q_maxs.append(agent.last_max_legal_q)

                    loss = agent.calculate_td_loss(
                        black_old_state, black_legal_mask, black_action,
                        black_reward, env.get_state(), done, env.action_mask(), collect
                    )

                    eval_losses.append(loss)
                    if collect:
                        td_target_values.append(agent.last_td_target)
                        td_abs_values.append(agent.last_td_abs)

            if is_agent_playing_for_white:
                if not is_self_play:
                    white_reward -= DISCOUNT_FACTOR_GAMMA * black_reward

                collect = current_agent_step % log_q_every == 0

                if is_self_play:
                    loss = agent.calculate_td_loss(
                        white_old_state, white_legal_mask, white_action,
                        white_reward, black_old_state, done, black_legal_mask, collect
                    )

                    eval_losses.append(loss)
                    if collect:
                        td_target_values.append(agent.last_td_target)
                        td_abs_values.append(agent.last_td_abs)

                else:
                    loss = agent.calculate_td_loss(
                        white_old_state, white_legal_mask, white_action,
                        white_reward, env.get_state(), done, env.action_mask(), collect
                    )

                    eval_losses.append(loss)
                    if collect:
                        td_target_values.append(agent.last_td_target)
                        td_abs_values.append(agent.last_td_abs)

        if not done:
            recent_outcomes["truncated"] += 1
        elif is_draw:
            recent_outcomes["draw"] += 1
        else:
            recent_outcomes["white_win" if is_white_on_turn else "black_win"] += 1

    process_stats(recent_outcomes, eval_losses, q_averages, q_maxs, q_mins, False,
                  is_agent_playing_for_white, is_agent_playing_for_black,
                  td_target_values, td_abs_values)

def process_stats(outcomes: Counter,
                  losses: list[float],
                  q_averages: list[float],
                  q_maxs: list[float],
                  q_mins: list[float],
                  is_training_stats: bool,
                  is_agent_playing_for_white: bool,
                  is_agent_playing_for_black: bool,
                  td_target_values: list[float] | None=None,
                  td_abs_values: list[float] | None=None) -> None:


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
    else:
        if is_agent_playing_for_white:
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
