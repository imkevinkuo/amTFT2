import random
import numpy as np


def valid_moves(i, Z=4):
    x, y = i % 5, i // 5
    return [(x < Z), (y < Z), (x > 0), (y > 0)]


def move_xy(i, move):
    if move == 0:
        i += 1
    if move == 1:
        i += 5
    if move == 2:
        i -= 1
    if move == 3:
        i -= 5
    return i


# def flip_uid(uid):
#     p0 = uid // 1250
#     p1 = (uid % 1250) // 50
#     coin = uid % 25
#     color = (uid % 50) // 25
#     return 1250 * p1 + 50 * p0 + coin + 25 * (1 - color)
#
#

# def uid_to_board(uid):
#     b = Board(None)
#     b.pos[0] = uid // 1250
#     b.pos[1] = (uid % 1250) // 50
#     b.coin = uid % 25
#     b.color = (uid % 50) // 25
#     return b


# Choosing triangle over own-colored coin
true_rewards = np.array([[[1, 0], [1, -2], [0, -5], [0, 0]],
                         [[-2, 1], [0, 1], [0, 0], [-5, 0]]])

self_rewards = np.array([[[1, 0], [1, 0], [0, 0], [0, 0]],
                         [[0, 1], [0, 1], [0, 0], [0, 0]]])
coop_rewards = np.array([[[1, 1], [-1, -1], [-5, -5], [0, 0]],
                         [[-1, -1], [1, 1], [0, 0], [-5, -5]]])
# p0 normal, p1 safe -> yes p0 will learn best response to "safe"
# safe_rewards = np.array([[[2, 0], [1, 0], [0, 0], [5, 0]],
#                          [[0, 1], [0, 2], [0, 5], [0, 0]]])
# p0 normal, p1 aggr -> p0 learns safe policy
safe_rewards = np.array([[[1, -1], [1, -1], [0, 0], [0, 0]],
                         [[-2, 2], [0, 0], [0, 0], [-5, 5]]])
# p0 aggr, p1 normal -> p0 learns aggr policy
aggr_rewards = np.array([[[0, 0], [2, -2], [5, -5], [0, 0]],
                         [[-1, 1], [-1, 1], [0, 0], [0, 0]]])

regimes = {
    "true": true_rewards,
    "self": self_rewards,
    "coop": coop_rewards,
    "safe": safe_rewards,
    "aggr": aggr_rewards
}


class Board:
    def __init__(self, rewards, pos=None):
        if pos is None:
            a = random.randrange(0, 26, 2)
            b = random.randrange(1, 25, 2)
        else:
            a, b = pos
        self.pos = [a, b]
        self.coins = [None, None]
        self.color = None
        self.picks = np.zeros((2, 4))  # defect / coop / safe / aggro
        self.prev_moves = [-1, -1]
        self.prev_uid = -1
        self.last_reward = np.zeros(2)
        self.total_reward = np.zeros(2)
        #
        self.rewards = true_rewards if rewards is None else rewards
        self.spawn_coin()

    def reset(self):
        # print(self.picks)
        a = random.randrange(0, 26, 2)
        b = random.randrange(1, 25, 2)
        self.pos = [a, b]
        self.coins = [None, None]
        self.color = None
        self.picks = np.zeros((2, 4))
        self.prev_moves = [-1, -1]
        self.prev_uid = -1
        self.last_reward = np.zeros(2)
        self.total_reward = np.zeros(2)
        self.spawn_coin()

    def copy(self):
        new_board = Board(self.rewards)
        new_board.pos = self.pos.copy()
        new_board.coins = self.coins.copy()
        new_board.color = self.color
        return new_board

    def uid(self):
        # c: 0-24 color 0, 25-49 color 1, 50 DNE
        c1 = 50 if self.coins[0] is None else self.coins[0] + 25 * self.color
        c2 = 25 if self.coins[1] is None else self.coins[1]
        uid = (25 * 25 * 50) * self.pos[0] + (25 * 50) * self.pos[1] + 50 * c2 + c1
        return uid

    def action_mask(self, agent):
        return [i for i, valid in enumerate(valid_moves(self.pos[agent])) if valid]

    def play_moves(self, next_moves):
        self.prev_uid = self.uid()
        self.pos[0] = move_xy(self.pos[0], next_moves[0])
        self.pos[1] = move_xy(self.pos[1], next_moves[1])
        self.prev_moves = next_moves
        return

    # both agents cannot simultaneously step on coin if we fix odd \ell_1 dist
    def check_reward(self):
        r = np.zeros(2)
        self.last_reward = np.zeros(2)
        for i in range(2):
            # Pick up malicious coin
            if self.pos[i] == self.coins[1]:
                self.coins[1] = None
                r += self.rewards[i][2 + self.color]
                self.last_reward += true_rewards[i][2 + self.color]
                self.total_reward += true_rewards[i][2 + self.color]
                if self.color != i:
                    self.picks[i][2] += 1  # safe
                else:
                    self.picks[i][3] += 1  # aggro
            # Regular coin
            if self.pos[i] == self.coins[0]:
                self.coins[0] = None
                r += self.rewards[i][self.color]
                self.last_reward += true_rewards[i][self.color]
                self.total_reward += true_rewards[i][self.color]
                if self.color != i:
                    self.picks[i][0] += 1  # defect
                else:
                    self.picks[i][1] += 1  # coop
        # if np.random.random() < 0.1:
        self.spawn_coin()
        return r

    # Spawns a coin in a non-occupied location.
    def spawn_coin(self):
        if self.coins[0] is None:
            self.coins[0] = random.choice([i for i in range(0, 25) if i not in self.pos])
            self.color = random.randint(0, 1)
        if self.coins[1] is None and np.random.random() > 0.9:
            self.coins[1] = random.choice([i for i in range(0, 25) if i not in self.pos and i not in self.coins])

    def display(self):
        print("WTF")
        # print(self.picks)
        grid = [['.' for _ in range(5)] for _ in range(5)]
        i, j = self.pos
        k, l = self.coins
        grid[i // 5][i % 5] = '0'
        grid[j // 5][j % 5] = '1'
        if k is not None:
            grid[k // 5][k % 5] = 'o' if self.color == 0 else 'i'
        if l is not None:
            grid[l // 5][l % 5] = 'v'
        for row in grid[::-1]:
            print("| " + ' '.join(row) + " |")
