"""
## Test
"""
seed = 42
model = keras.models.load_model('/content/drive/MyDrive/game_ai/model', compile=False)

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

env = gym.wrappers.Monitor(
    env,
    '/content/drive/MyDrive/game_ai/video',
    video_callable=lambda episode_id: True,
    force=True
)

n_episodes = 10
epsilon = 0
rewards = np.zeros(n_episodes, dtype=float)

for i in range(n_episodes):
    # Resetting the state for each episode
    state = np.array(env.reset())
    done = False

    while not done:
        # Initiate greedy policy
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_values = model.predict(state_tensor)
        action = np.argmax(action_values)

        # Perform action
        state_next, reward, done, _ = env.step(action)
        # Progress to next state
        state = np.array(state_next)

        # Store the reward
        rewards[i] += reward

env.close()

print('Returns: {}'.format(rewards))
