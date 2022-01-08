import numpy as np

model_free = __import__('3_tabular_model_free_rl')


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    # Number of actions
    num_actions = env.n_actions

    for i in range(max_episodes):
        state = env.reset()

        # Q-value
        q_value = state.dot(theta)

        # Choose optimum action using epsilon-greedy policy
        current_action = model_free.epsilon_greedy(q_value, epsilon[i], num_actions, random_state)

        terminal = False
        while not terminal:
            next_state, reward, terminal = env.step(current_action)

            # Store temporal difference
            delta = reward - q_value[current_action]

            q_value = next_state.dot(theta)

            # Selecting the next action according to epsilon-greedy policy
            new_action = model_free.epsilon_greedy(q_value, epsilon[i], num_actions, random_state)

            # Update values according to the new action (a')
            delta += gamma * q_value[new_action]
            theta += eta[i] * delta * state[current_action]
            state = next_state
            current_action = new_action

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    # Number of actions
    num_actions = env.n_actions

    for i in range(max_episodes):
        state = env.reset()
        q_value = state.dot(theta)

        terminal = False
        while not terminal:
            # Choose optimum action using epsilon-greedy policy
            current_action = model_free.epsilon_greedy(q_value, epsilon[i], num_actions, random_state)
            next_state, reward, terminal = env.step(current_action)

            # Store temporal difference
            delta = reward - q_value[current_action]

            # Calculate new Q-values for the next state
            q_value = next_state.dot(theta)

            # Update values according to the new action (a')
            delta += gamma * max(q_value)
            theta += eta[i] * delta * state[current_action]
            state = next_state

    return theta
