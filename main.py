import random

import agents
import coins
import coins_aec
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


configs = []
# for regime in ['self', 'coop', 'aggr']:
for regime in ['coop']:
    for lr in [0.8]:
        for y in [0.95]:
            configs.append({
                "regime": regime,
                "lr": lr,
                "y": y
            })


tables = {}
for obs in ["", "_true"]:
    for regime in ["self", "coop", "aggr", "safe"]:
        for agent in ["player_0", "player_1"]:
            trials_Q = np.stack([np.load(f"qtables/{agent + obs}-regime_{regime}-lr_0.8-y_0.95-trial_{i}.npy") for i in range(5)], axis=0)
            tables[(agent + obs, regime)] = trials_Q[0]


def train(opponent=None):
    results = []
    for config in configs:
        regime, lr, y = config["regime"], config["lr"], config["y"]
        histories = []
        for trial in range(5):
            env = coins_aec.env(coins.regimes[regime])
            env.reset()
            true_env = coins_aec.env()
            true_env.reset()
            if opponent is None:
                learners, history = agents.train_base_agents(env, lr, y, episodes=50000)
            else:
                learners, history = agents.train_best_response(env, opponent, lr, y, episodes=50000)
            for agent, learner in learners.items():
                np.save(f"{agent}-regime_{regime}-lr_{lr}-y_{y}-trial_{trial}.npy", learner.Q)

            env.reset()
            histories.append(history)

        results.append(np.array(histories))

    return results


def test(p0, p1, tables):
    env = coins_aec.env()

    if p0 == "amTFT":
        p0 = agents.amTFT(tables)
    elif p0 == "amTFT*":
        p0 = agents.amTFT(tables, safe=True)
    else:
        p0 = agents.Greedy(0, 0, Q=tables[("player_0", p0)])

    if p1 == "human":
        p1 = agents.Human()
    elif p1 == "amTFT":
        p1 = agents.amTFT(tables, p=1)
    elif p1 == "amTFT*":
        p1 = agents.amTFT(tables, p=1, safe=True)
    else:
        p1 = agents.Greedy(0, 0, Q=tables[("player_1", p1)])

    env.reset()
    A = {"player_0": p0, "player_1": p1}
    for agent in env.agent_iter():
        if agent == "player_0" and p1 == "human":
            env.render()
            print()
        observation, reward, done, info = env.last()
        A[agent].observe(observation, reward)
        action = A[agent].policy(observation)
        env.step(action)
    return observation['observation'].picks


# Plot results
def plot_train(results, smooth=False):
    for i in range(len(results)):
        histories = results[i]
        config = configs[i]
        regime, lr, y = config["regime"], config["lr"], config["y"]

        # histories[trial, episode, agent, action_type] = frequency
        action_strs = ["defect", "coop", "safe", "aggro"]
        for agent_idx in range(2):
            fig, ax = plt.subplots(1, 1, figsize=(8, 12))
            for action in range(4):
                traj = histories.mean(axis=0)[:, agent_idx, action]
                if smooth:
                    window_width = 12
                    cumsum_vec = np.cumsum(np.insert(traj, 0, 0))
                    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
                    ax.plot(ma_vec, label=action_strs[action])
                    ax.tick_params(axis='x', labelsize=24)
                    ax.tick_params(axis='y', labelsize=24)
                    ax.set_ylim(0, 35)
                else:
                    ax.plot(traj, label=action_strs[action])
            # ax.set_title(f"agent-{agent_idx}-{regime}-{lr}-{y}")
            ax.legend(loc=2, prop={'size': 24})
            plt.tight_layout()
            fig.show()


def eval1(seed=0):
    agents = ["self", "coop", "amTFT", "aggr", "safe", "amTFT*"]
    matchups = [(i, j) for i in agents for j in agents]
    # matchups = [("self", "amTFT"), ("coop", "amTFT")]
    matchup_scores = []
    matchup_stdevs = []
    trials = 100
    for matchup in tqdm(matchups):
        np.random.seed(seed)
        random.seed(seed)
        # print(matchup)
        a, b = matchup
        results = np.array([test(a, b, tables) for i in range(trials)])

        p0_points = np.array([[1, 1, 0, 0],
                              [-2, 0, 0, -5]])
        p1_points = np.array([[-2, 0, 0, -5],
                              [1, 1, 0, 0]])

        trial_scores = (results*p0_points).reshape(trials, -1).sum(axis=1)
        mean = np.mean(trial_scores, axis=0)
        stdev = np.std(trial_scores, axis=0)

        # print(results.mean(axis=0))
        # print(mean)

        matchup_scores.append(mean)
        matchup_stdevs.append(stdev)

    latex(matchup_scores, matchup_stdevs)
    return matchup_scores, matchup_stdevs


def latex(matchup_scores, matchup_stdevs):
    # paste into latex
    agent_names = ["Defect (D)", "Cooperate (C)", "amTFT", "Aggressive (A)", "Safe (S)", "amTFT*"]
    matchup_scores = np.array(matchup_scores).reshape(6, 6)
    final_scores = matchup_scores.copy()
    # "safe" regime = p2 is aggressive
    final_scores[:, 3] = matchup_scores[:, 4]
    final_scores[:, 4] = matchup_scores[:, 3]

    matchup_stdevs = np.array(matchup_stdevs).reshape(6, 6)
    final_stdevs = matchup_stdevs.copy()
    # "safe" regime = p2 is aggressive
    final_stdevs[:, 3] = matchup_stdevs[:, 4]
    final_stdevs[:, 4] = matchup_stdevs[:, 3]

    for i in range(len(final_scores)):
        sc = final_scores[i]
        st = final_stdevs[i]
        print("\\hline")
        print(agent_names[i].ljust(14) + " & " + ' & '.join((f"{sc[j]:.1f} ({st[j]:.1f})".rjust(6, ' ') for j in range(len(sc)))), "\\\\")

    print()
    stat_names = ["Self-match", "Selfish safety", "Minmax safety", "IncentC"]
    stat_formulas = ["$S_1(X, X)$", "$S_1(X, D) - S_1(D, D)$", "$S_1(X, A) - S_1(S, A)$", "$S_2(X, C) - S_2(X, D)$"]
    stats = []
    stats.append([final_scores[i, i] for i in range(6)])
    stats.append([final_scores[i, 0] - final_scores[0, 0] for i in range(6)])
    stats.append([final_scores[i, 3] - final_scores[4, 3] for i in range(6)])
    stats.append([final_scores[1, i] - final_scores[0, i] for i in range(6)])
    stats = np.array(stats)
    for i, arr in enumerate(stats):
        print("\\hline")
        print(stat_names[i] + ' & ' + stat_formulas[i] + ' & ' + ' & '.join((f"{j:.1f}".rjust(6, ' ') for j in arr)),
              "\\\\")


# results = train()
# plot_train(results, True)

