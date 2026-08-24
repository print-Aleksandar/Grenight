import os
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import Counter
from agents.q_network import QNetwork
from agents.worker import worker_loop
from agents.double_dqn_vs_random_agent import DoubleDQNAgent
from domain.configs import (
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    CHECKPOINT_EVERY_EPISODES,
    LOG_EVERY_EPISODES,
    CHECKPOINT_DIR
)
from environment.grenight_environment import GrenightEnvironment

AGENT_IS_WHITE = True
SYNC_EVERY_GLOBAL_STEPS = 500
TOTAL_EPISODES = 50_000


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"will save checkpoints in: {CHECKPOINT_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    ref_env = GrenightEnvironment()
    num_actions = ref_env.action_encoder.NUM_ACTIONS
    num_planes = ref_env.state_encoder.NUM_PLANES

    agent = DoubleDQNAgent(
        num_planes=num_planes, rows=5, columns=4,
        num_actions=num_actions, device=device
    )
    print(f"policy_net device: {next(agent.policy_net.parameters()).device}")

    shared_policy_net = QNetwork(num_planes, 5, 4, num_actions)
    shared_policy_net.load_state_dict(
        {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
    )
    shared_policy_net.eval()
    shared_policy_net.share_memory()

    transition_queue = mp.Queue(maxsize=20_000)
    episode_queue = mp.Queue(maxsize=5_000)
    stop_event = mp.Event()

    num_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"spawning {num_workers} worker processes")

    workers = []
    for _ in range(num_workers):
        p = mp.Process(
            target=worker_loop,
            args=(shared_policy_net, transition_queue, episode_queue, stop_event,
                  EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS,
                  num_actions, AGENT_IS_WHITE),
        )
        p.start()
        workers.append(p)

    global_step = 0
    episode = 0
    losses = []
    episode_lengths = []
    recent_outcomes = Counter()

    def save_checkpoint(ep: int, tag: str = ""):
        path = os.path.join(CHECKPOINT_DIR, f"double_dqn_vs_random_ep{ep}{tag}.pt")
        torch.save({
            "episode": ep,
            "global_step": global_step,
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        }, path)
        print(f"[checkpoint] saved: {path}")

    def check_q_values():
        if len(agent.replay_buffer) < 500:
            return None
        batch = agent.replay_buffer.sample(500)
        states_t = torch.from_numpy(np.stack([t.state for t in batch])).to(device)
        with torch.no_grad():
            q = agent.policy_net(states_t)
        return q.min().item(), q.max().item(), q.mean().item()

    try:
        while episode < TOTAL_EPISODES:

            drained = 0
            while not transition_queue.empty() and drained < 2000:
                state, action, reward_white, next_state, done, next_legal_mask, next_is_white_turn = transition_queue.get()
                agent.store(state, action, reward_white, next_state, done, next_legal_mask, next_is_white_turn)
                global_step += 1
                drained += 1

            if drained > 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                if global_step % SYNC_EVERY_GLOBAL_STEPS < drained:
                    shared_policy_net.load_state_dict(
                        {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
                    )

            while not episode_queue.empty():
                move_count, outcome, local_agent_step = episode_queue.get()
                episode += 1
                episode_lengths.append(move_count)
                recent_outcomes[outcome] += 1

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
                    total_eps = completed + recent_outcomes.get("truncated", 0)

                    agent_win_pct = 100.0 * recent_outcomes.get("agent_win", 0) / completed if completed > 0 else 0.0
                    random_win_pct = 100.0 * recent_outcomes.get("random_win", 0) / completed if completed > 0 else 0.0
                    draw_pct = 100.0 * recent_outcomes.get("draw", 0) / completed if completed > 0 else 0.0
                    truncated_pct = 100.0 * recent_outcomes.get("truncated", 0) / total_eps if total_eps > 0 else 0.0

                    print(
                        f"ep={episode:>7} "
                        f"step={global_step:>8} "
                        f"avg_len={avg_len:>6.1f} "
                        f"avg_loss={avg_loss:>8.4f} "
                        f"agent_win={agent_win_pct:5.1f}% "
                        f"random_win={random_win_pct:5.1f}% "
                        f"draw={draw_pct:5.1f}% "
                        f"truncated={truncated_pct:5.1f}% "
                        f"outcomes(last{LOG_EVERY_EPISODES})={dict(recent_outcomes)}"
                    )
                    recent_outcomes.clear()

                    q_stats = check_q_values()
                    if q_stats is not None:
                        q_min, q_max, q_mean = q_stats
                        print(f"        Q min={q_min:8.3f}  Q max={q_max:8.3f}  Q mean={q_mean:8.3f}")

            if drained == 0 and episode_queue.empty():
                continue

    except KeyboardInterrupt:
        print("\n[interrupted] saving checkpoint before exit...")
        save_checkpoint(episode, tag="_interrupted")

    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=5)
        save_checkpoint(episode, tag="_final")
        print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
    