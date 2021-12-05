import numpy as np
import coins
from tqdm import tqdm


def update_Q(Q, lr, y, s, a, new_s, reward):
    Q[s, a] = Q[s, a] + lr * (reward + y * np.max(Q[new_s, :]) - Q[s, a])


class amTFT:
    # Q is schedule-specific reward. V is the real game reward from following the schedule-learned policies.
    def __init__(self, tables, p=0, a=1, T=1, safe=False):
        self.Q1_CC = tables[("player_0", "coop")]
        self.Q2_CC = tables[("player_1", "coop")]
        self.Q1_DD = tables[("player_0", "self")]
        self.Q2_DD = tables[("player_1", "self")]
        self.V1_CC = tables[("player_0_true", "coop")]
        self.V2_CC = tables[("player_1_true", "coop")]
        self.V1_DD = tables[("player_0_true", "self")]
        self.V2_DD = tables[("player_1_true", "self")]
        self.Q1_SA = tables[("player_0", "safe")]
        self.b = 0
        self.debit = 0
        self.danger = 0
        self.a = a
        self.T = T
        self.p = p
        self.safe = safe  # amTFT*
        if p == 1:
            self.Q1_CC = tables[("player_1", "coop")]
            self.Q2_CC = tables[("player_0", "coop")]
            self.Q1_DD = tables[("player_1", "self")]
            self.Q2_DD = tables[("player_0", "self")]
            self.V1_CC = tables[("player_1_true", "coop")]
            self.V2_CC = tables[("player_0_true", "coop")]
            self.V1_DD = tables[("player_1_true", "self")]
            self.V2_DD = tables[("player_0_true", "self")]
            self.Q1_SA = tables[("player_1", "aggr")]
        self.prev_state = -1

    def policy(self, observation):
        board = observation["observation"]
        state = board.uid()
        mask = observation['action_mask']

        if self.b == 0:
            Q = self.Q1_CC
        elif self.b == -1:
            Q = self.Q1_SA
        else:
            Q = self.Q1_DD

        subset_idx = np.argmax((Q[state] + np.random.random(4))[mask])
        parent_idx = np.arange(Q.shape[1])[mask][subset_idx]
        action = parent_idx
        return action

    def observe(self, observation, reward):
        board = observation["observation"]
        s = board.prev_uid
        a2 = board.prev_moves[1]
        if self.b == 0:
            # amTFT coop-defect switching rule
            C2 = np.argmax(self.Q2_CC[s])
            D = self.V2_CC[s, a2] - self.V2_CC[s, C2]
            self.debit += max(0, D)
            # print(self.debit)
            if self.debit > self.T:
                self.b = self.defect_steps(observation)
                self.debit = 0
                self.a += 1
        elif self.b > 0:
            # print("Defecting for", self.b, "turns")
            self.b -= 1
            if self.safe:
                # amTFT* defect-safe switching rule
                D = np.max(self.V1_DD[s]) - np.max(self.V1_DD[board.uid()])
                if D > self.T:
                    self.b = -1

    def defect_steps(self, observation):
        B = 100  # number of replicates to simulate
        M = 20  # max number of turns to roll out
        board = observation["observation"]
        replicates = {
            "cc": [board.copy() for b in range(B)],
            "dd": [board.copy() for b in range(B)]
        }
        C1 = Greedy(0, 0, Q=self.Q1_CC)
        C2 = Greedy(0, 0, Q=self.Q2_CC)
        D1 = Greedy(0, 0, Q=self.Q1_DD)
        D2 = Greedy(0, 0, Q=self.Q2_DD)
        for m in range(M):
            for bcc in replicates["cc"]:
                bcc.play_moves((C1.policy_s(bcc.uid(), bcc.action_mask(0)),
                                C2.policy_s(bcc.uid(), bcc.action_mask(1))))
                bcc.check_reward()
            for bdd in replicates["dd"]:
                bdd.play_moves((D1.policy_s(bdd.uid(), bdd.action_mask(0)),
                                D2.policy_s(bdd.uid(), bdd.action_mask(1))))
                bdd.check_reward()
            mean_cc = np.mean([bcc.total_reward for bcc in replicates["cc"]], axis=0)
            mean_dd = np.mean([bdd.total_reward for bdd in replicates["dd"]], axis=0)
            # print(mean_cc, mean_dd, mean_cc - mean_dd)
            if (mean_cc - mean_dd)[1 - self.p] > self.a * self.T:
                return m
        return M


class Greedy:
    def __init__(self, num_states, num_actions, lr=0.8, y=0.95, Q=None):
        self.Q = np.zeros((num_states, num_actions)) if Q is None else Q
        self.lr = lr
        self.y = y
        self.prev_state = -1
        self.prev_action = -1

    def policy_s(self, state, mask, epsilon=0.0, beta=1.0):
        # print(self.Q[state])
        subset_idx = np.argmax((beta*self.Q[state] + np.random.random(4))[mask])
        parent_idx = np.arange(self.Q.shape[1])[mask][subset_idx]
        action = parent_idx
        # if np.random.random() > epsilon:
        #     subset_idx = np.argmax((self.Q[state])[mask])
        #     parent_idx = np.arange(self.Q.shape[1])[mask][subset_idx]
        #     action = parent_idx
        # else:
        #     action = mask[np.random.randint(0, len(mask))]
        self.prev_action = action
        return action

    def policy(self, observation, epsilon=0.0, beta=1.0):
        return self.policy_s(observation['observation'].uid(), observation['action_mask'], epsilon, beta)

    def observe(self, observation, reward, update=False):
        new_state = observation["observation"].uid()
        if self.prev_state != -1 and self.prev_action != -1 and update:
            update_Q(self.Q, self.lr, self.y, self.prev_state, self.prev_action, new_state, reward)
        self.prev_state = new_state


class Human:
    def __init__(self):
        i = 0

    def policy(self, observation):
        valid = False
        action = -1
        while not valid:
            action = int(input(f"Choose move: "))
            valid = action in observation['action_mask']
        return action

    def observe(self, observation, reward):
        i = 0


def train_base_agents(env, lr=0.8, y=0.95, episodes=1000):
    agent_0 = env.possible_agents[0]
    num_states = env.observation_space(agent_0)["observation"].n
    num_actions = env.action_space(agent_0).n
    obs = np.zeros(num_states)
    obs_max = 0
    learners = {}
    for agent in env.possible_agents:
        learners[agent] = Greedy(num_states, num_actions, lr, y)  # Contains Q-table for policy
        learners[agent + "_true"] = Greedy(num_states, num_actions, lr, y)  # Contains true Q-table
    history = []
    for episode in tqdm(range(episodes)):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            learners[agent].observe(observation, reward, update=True)
            # obs_frac = obs[observation['observation'].uid()] / obs_max if obs_max > 0 else 0
            # obs_frac = 0
            action = learners[agent].policy(observation, beta=episode/episodes)
            learners[agent + "_true"].observe(observation, info, update=True)
            learners[agent + "_true"].prev_action = action
            env.step(action)
        obs[observation['observation'].uid()] += 1
        obs_max = max(obs[observation['observation'].uid()], obs_max)
        history.append(observation["observation"].picks)
    return learners, np.array(history)


# def train_best_response(env, opponent, lr=0.8, y=0.95, episodes=1000):
#     agent_0 = env.possible_agents[0]
#     agent_1 = env.possible_agents[1]
#     num_states = env.observation_space(agent_0)["observation"].n
#     num_actions = env.action_space(agent_0).n
#     obs = np.zeros(num_states)
#     obs_max = 0
#     learners = {}
#     for agent in env.possible_agents:
#         if agent == agent_0:
#             learners[agent] = opponent
#             learners[agent + "_true"] = opponent
#         else:
#             learners[agent] = Greedy(num_states, num_actions, lr, y)  # Contains Q-table for policy
#             learners[agent + "_true"] = Greedy(num_states, num_actions, lr, y)  # Contains true Q-table
#     history = []
#     for episode in tqdm(range(episodes)):
#         env.reset()
#         for agent in env.agent_iter():
#             observation, reward, done, info = env.last()
#             learners[agent].observe(observation, reward, update=agent == agent_1)
#             # epsilon = (1 - (episode / episodes)) if agent == agent_1 else 0
#             beta = episode / episodes if agent == agent_1 else 1
#             action = learners[agent].policy(observation, beta=beta)
#             learners[agent + "_true"].observe(observation, info, update=agent == agent_1)
#             learners[agent + "_true"].prev_action = action
#             env.step(action)
#         obs[observation['observation'].uid()] += 1
#         obs_max = max(obs[observation['observation'].uid()], obs_max)
#         history.append(observation["observation"].picks)
#     return learners, np.array(history)
