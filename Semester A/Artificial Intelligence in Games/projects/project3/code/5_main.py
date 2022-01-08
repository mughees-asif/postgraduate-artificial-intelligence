import numpy as np
from matplotlib import pyplot as plt

environment = __import__('1_environment')
t_model_based = __import__('2_tabular_model_based_rl')
t_model_free = __import__('3_tabular_model_free_rl')
nt_model_free = __import__('4_non_tabular_model_free_rl')

if __name__ == '__main__':
    seed = 0
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = environment.FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print('\n# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value, iterations = t_model_based.policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    # print('\n---\nPolicy iterations: {:d}\n---'.format(iterations))

    print('')

    print('## Value iteration')
    policy, value, iterations = t_model_based.value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    # print('\n---\nValue iterations: {:d}\n---'.format(iterations))

    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value = t_model_free.sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    fig, ab = plt.subplots()
    ab.plot(value)
    ab.set_title('SARSA')
    ab.set_xlabel('Steps')
    _ = ab.set_ylabel('Value')
    plt.show()
    print('')

    print('## Q-learning')
    policy, value = t_model_free.q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    fig, ab = plt.subplots()
    ab.plot(value)
    ab.set_title('Q-learning')
    ab.set_xlabel('Steps')
    _ = ab.set_ylabel('Value')
    plt.show()
    print('')

    linear_env = nt_model_free.LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = nt_model_free.linear_sarsa(linear_env, max_episodes, eta,
                                            gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = nt_model_free.linear_q_learning(linear_env, max_episodes, eta,
                                                 gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
