from gym import spaces
from gym.spaces import Discrete
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from coins import Board


def env(rewards=None):
    env = raw_env(rewards)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {'render.modes': ['human'], "name": "coins"}

    def __init__(self, rewards):
        self.board = Board(rewards)
        self.agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = self.agents
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.action_spaces = {agent: Discrete(4) for agent in self.agents}
        self.observation_spaces = {i: spaces.Dict({
            'observation': spaces.Discrete(25*25*25*51),
            'action_mask': spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)
        }) for i in self.agents}

    def render(self, mode="human"):
        self.board.display()

    def observe(self, agent):
        return {'observation': self.board, 'action_mask': self.board.action_mask(self.agents.index(agent))}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        pass

    def reset(self):
        # print(self.board.coops)
        # print(self.board.defects)
        # print()
        self.board.reset()
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: 0 for agent in self.agents}  # info will store true reward regardless of regime
        self.next_actions = {agent: -1 for agent in self.agents}
        self.observations = {agent: self.board for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(None)  # (action) ??

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0
        self.next_actions[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.board.play_moves([self.next_actions[agent] for agent in self.agents])

            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = self.board.check_reward()
            self.infos[self.agents[0]], self.infos[self.agents[1]] = self.board.last_reward

            self.num_moves += 1

            # NUM_ITERS = 0 if np.random.random() >= 0.998 else 10000
            NUM_ITERS = 500
            self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.board
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
