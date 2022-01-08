import numpy as np

model_based = __import__('2_tabular_model_based_rl')


# Define epsilon-greedy policy to balance exploration and exploitation
def epsilon_greedy(q, num_actions, epsilon, random_state):
    if random_state.uniform(0, 1) < epsilon:
        return random_state.choice(np.flatnonzero(q == q.max()))
    else:
        return random_state.randint(num_actions)


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # Number of actions
    num_actions = env.n_actions
    # Q-value
    q_value = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        # Reset state
        state = env.reset()
        # Select the action according to the epsilon-greedy policy
        action = epsilon_greedy(q_value[state], num_actions, epsilon[i], random_state)

        terminal = False
        while not terminal:
            next_state, reward_current, terminal = env.step(action)
            # Selecting the next action according to epsilon-greedy policy
            next_action = epsilon_greedy(q_value[next_state], num_actions, epsilon[i], random_state)
            # Store the Q-values denoting desirability of the action
            q_value[state, action] += eta[i] * (
                    reward_current + (gamma * q_value[next_state, next_action]) - q_value[state, action]
            )
            # Replace the current state with the next state
            state = next_state
            # Replace the current action with the next action
            action = next_action

        policy = q_value.argmax(axis=1)
        value = q_value.max(axis=1)
        value_next = model_based.policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100)
        if all(abs(value_next[count] - value[count]) < 0.1 for count in range(len(value_next))):
            return policy, value_next

    policy = q_value.argmax(axis=1)
    value = q_value.max(axis=1)
    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # Number of actions
    num_actions = env.n_actions
    # Q-value
    q_value = np.zeros((env.n_states, num_actions))

    for i in range(max_episodes):
        # Reset state
        state = env.reset()

        terminal = False
        while not terminal:
            # Selecting the next action according to epsilon-greedy policy
            action = epsilon_greedy(q_value[state], num_actions, epsilon[i], random_state)
            next_state, reward_current, terminal = env.step(action)
            # Store the Q-values denoting desirability of the action
            q_value[state, action] += eta[i] * (
                    reward_current + (gamma * max(q_value[next_state])) - q_value[state, action]
            )
            state = next_state  # storing the next state as the current state for the next episode

        policy = q_value.argmax(axis=1)
        value = q_value.max(axis=1)
        value_next = model_based.policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100)
        if all(abs(value_next[count] - value[count]) < 0.1 for count in range(len(value_next))):
            return policy, value_next

    policy = q_value.argmax(axis=1)
    value = q_value.max(axis=1)
    return policy, value
