import numpy as np
import contextlib
from itertools import product


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):

    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        self.random_state = np.random.RandomState(seed)

        n_states = self.lake.size + 1
        self.n_states = n_states
        self.absorbing_state = n_states - 1

        self.n_actions = 4
        self.max_steps = max_steps

        # Available actions
        self.actions = [
            (0, 1),  # Up
            (-1, 0),  # Left
            (0, -1),  # Down
            (1, 0)  # Right
        ]

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.pi = pi

        # Develop reward map with the following values:
        # 1: Goal
        # 0: No goal
        matrix = np.array(lake)
        reward_map = np.zeros([len(matrix), len(matrix[0])])
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] != '$':
                    reward_map[i, j] = 0  # != goal
                else:
                    reward_map[i, j] = 1  # == goal

        self.reward_map = reward_map

        # Compute the transition probabilities
        self.transition_probs = np.zeros((self.n_states, self.n_states, self.n_actions))

        # State-action pairs
        self.state_action_pairs = list(product(range(self.reward_map.shape[0]), range(self.reward_map.shape[1])))
        self.state_action_pairs.append(tuple((-1, -1)))

        # Map probabilities to rewards
        self.probs_to_rewards = {s: i for (i, s) in enumerate(self.state_action_pairs)}

        # Iterate over the state space i.e., the grid
        for state_idx, state in enumerate(self.state_action_pairs):
            for action_idx, action in enumerate(self.actions):
                # Reached absorb state
                if state_idx == self.absorbing_state:
                    next_state_idx = self.probs_to_rewards.get(next_state, state_idx)
                    # Maintain absorb state
                    self.transition_probs[next_state_idx, state_idx, action_idx] = 1.0

                #
                elif state_idx != self.absorbing_state and \
                        (self.lake_flat[state_idx] == '#' or
                         self.lake_flat[state_idx] == '$'):
                    next_state = (-1, -1)
                    next_state_idx = self.probs_to_rewards.get(next_state, state_idx)
                    self.transition_probs[next_state_idx, state_idx, action_idx] = 1.0
                else:
                    # Update position
                    next_state = (state[0] + action[0], state[1] + action[1])

                    # Store current state
                    states = []
                    for i in range(4):
                        # Consider all slip directions
                        s = (state[0] + self.actions[i][0], state[1] + self.actions[i][1])
                        states.append(s)

                    # Keep track of future states
                    next_states = []
                    # Default to current state index
                    next_state_idx = self.probs_to_rewards.get(next_state, state_idx)
                    for i in range(4):
                        # Update to the next state based on the slip direction
                        s1 = self.probs_to_rewards.get(states[i], state_idx)
                        next_states.append(s1)

                    # define a probability
                    for i in range(4):
                        # Probability to slip given 4 possible directions
                        self.transition_probs[next_states[i], state_idx, action_idx] += self.slip / 4

                    # Update final probabilities
                    self.transition_probs[next_state_idx, state_idx, action_idx] += 1.0 - self.slip

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def act(self, state):
        return self.actions

    def p(self, next_state, state, action):
        return self.transition_probs[next_state, state, action]

    def r(self, next_state, state, action):
        if state != self.absorbing_state:
            return self.reward_map[self.state_action_pairs[state]]
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            actions = ['^', '_', '<', '>']

            print('Lake: ')
            print(self.lake)

            print('Policy: ')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    def play(env):
        actions = ['w', 'a', 's', 'd']

        state = env.reset()
        env.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid Action')

            state, r, done = env.step(actions.index(c))

            env.render()
            print('Reward: {0}.'.format(r))
