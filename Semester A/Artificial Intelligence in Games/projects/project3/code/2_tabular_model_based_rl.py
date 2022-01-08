import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    # Number of states
    num_states = env.n_states
    # The transition probability from current state to the next
    transition_probs = env.p
    # The transition reward from current state to the next
    transition_rewards = env.r

    for _ in range(max_iterations):
        delta = 0
        for state in range(num_states):
            # Value of the current state
            value_current = value[state]
            value[state] = sum([
                # The sum of probabilities
                transition_probs(next_state, state, policy[state]) *
                # Reward + (Discount factor) * Value
                (transition_rewards(next_state, state, policy[state]) + gamma * value[next_state])
                for next_state in range(num_states)
            ])
            # Delta = max(current delta, absolute maximum change in the value)
            delta = max(delta, abs(value_current - value[state]))
        if delta < theta:
            break
    return value


def policy_improvement(env, policy, value, gamma):
    # Number of states
    num_states = env.n_states
    # The transition probability from current state to the next
    transition_probs = env.p
    # The transition reward from current state to the next
    transition_rewards = env.r

    # Initialise the policy
    if policy is None:
        policy = np.zeros(num_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    policy_stable = True
    for state in range(num_states):
        # Store copy of the current policy
        policy_current = policy[state].copy()

        actions = []
        # Track index of each action
        for action_idx in range(len(env.act(state))):
            actions.append(action_idx)

        # Maintain track of actions with corresponding policy
        policy[state] = actions[int(
            # Maximum value for all actions
            np.argmax([
                # Sum probabilities for all possible next states
                sum([
                    transition_probs(next_state, state, action) *
                    (transition_rewards(next_state, state, action) + gamma * value[next_state])
                    for next_state in range(num_states)
                ])
                for action in actions])
        )]
        # If the policy changes, label as unstable
        if policy_current != policy[state]:
            policy_stable = False

    return policy, policy_stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    # Number of states
    num_states = env.n_states
    value = np.zeros(num_states, dtype=int)

    # Initialise the policy
    if policy is None:
        policy = np.zeros(num_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # Track iterations
    iterations = 0

    # Iterate till unstable policy
    policy_stable = True
    while policy_stable:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, policy_stable = policy_improvement(env, policy, value, gamma)
        iterations += 1

    return policy, value, iterations


def value_iteration(env, gamma, theta, max_iterations, value=None):
    # Number of states
    num_states = env.n_states
    # The transition probability from current state to the next
    transition_probs = env.p
    # The transition reward from current state to the next
    transition_rewards = env.r

    # Initialise the value
    if value is None:
        value = np.zeros(env.n_states)  # array of zeros size of n_states
    else:
        value = np.array(value, dtype=np.float)  # array size of value, type numpy float

    iterations = 0

    for _ in range(max_iterations):
        delta = 0
        for state in range(num_states):
            actions = []
            # Track index of each action
            for action_idx in range(len(env.act(state))):
                actions.append(action_idx)

            # Value of the current state
            value_current = value[state]
            value[state] = max([
                sum([
                    transition_probs(next_state, state, action) *
                    (transition_rewards(next_state, state, action) + gamma * value[next_state])
                    for next_state in range(num_states)])
                for action in actions])
            delta = max(delta, abs(value_current - value[state]))
        if delta < theta:
            break
        iterations += 1

    # Initialise policy
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):  # for all current state in n_states
        actions = []
        # Track index of each action
        for action_idx in range(len(env.act(state))):
            actions.append(action_idx)

        # Maintain track of actions with corresponding policy
        policy[state] = actions[int(
            # Maximum value for all actions
            np.argmax([
                # Sum probabilities for all possible next states
                sum([
                    transition_probs(next_state, state, action) *
                    (transition_rewards(next_state, state, action) + gamma * value[next_state])
                    for next_state in range(num_states)
                ])
                for action in actions])
        )]

    return policy, value, iterations
