import gym
import numpy as np
import tensorflow as tf

from utils import OUActionNoise, get_actor, get_critic, policy


def test(env_info, total_episodes=3, noise_std=0.2):
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std) * np.ones(1))

    for _ in range(total_episodes):

        prev_state = env.reset()

        while True:
            env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy(actor_model, tf_prev_state, ou_noise, **env_info)
            state, reward, done, info = env.step(action)

            if done:
                break
            prev_state = state


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    env_info = {
        "num_states": env.observation_space.shape[0],
        "num_actions": env.action_space.shape[0],
        "upper_bound": env.action_space.high[0],
        "lower_bound": env.action_space.low[0],
    }

    actor_model = get_actor(**env_info)
    critic_model = get_critic(**env_info)
    target_actor = get_actor(**env_info)
    target_critic = get_critic(**env_info)
    actor_model.load_weights(".model/pendulum_actor-100.h5")
    critic_model.load_weights(".model/pendulum_critic-100.h5")
    target_actor.load_weights(".model/pendulum_target_actor-100.h5")
    target_critic.load_weights(".model/pendulum_target_critic-100.h5")

    test(env_info)
