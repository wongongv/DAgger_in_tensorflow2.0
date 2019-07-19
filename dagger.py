import tensorflow as tf
from tensorflow.keras import layers
import argparse
import worker
from absl import app
import gym
import replay_buffer
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-ep','--episode', default = 100, type = int)
parser.add_argument('-i','--iter', default = 10, type = int)
parser.add_argument('-n', '--max_step', default = 10000, type = int)
parser.add_argument('-tr', '--training', default = 100, type = int)
parser.add_argument('--env', default = 'BipedalWalker-v2', type = str)
parser.add_argument('--exp_name', default = 'my', type = str)
parser.add_argument('--expert_file', default = './experts/test2.h5', type = str)
parser.add_argument('--save_dir', default = 'my_dir', type = str)
parser.add_argument('--load', default = 'False', action = 'store_true')


# Dagger






def main(args):
    #load expert_policy

    #It depends on
    check_dir = "./ppo2"
    expert_model = tf.keras.models.load_model(args.expert_file)


    #considering improve efficiency of replay buffer. (maybe set size etc)
    rep_buffer = replay_buffer.Buffer()

    #make env. here, using gym to test
    env = gym.make(args.env)

    #agent
    agent = worker.Agent(env)

    #initial sampling of expert trajectories
    expert_trajectories = agent.sample_trajectories(expert_model, env)


    #put above into replay_buffer
    rep_buffer.push(expert_trajectories)


    for ep in range(args.episode):

        # training
        for _ in range(args.training):
            rep_buffer.shuffle()
            samples = rep_buffer.sample()
            obs = samples[0][:,: env.observation_space.shape[0]]
            acs = samples[0][:,env.observation_space.shape[0] :]
            agent.train(obs, acs)


        obs = env.reset()
        states_to_be_labled = np.expand_dims(np.copy(obs), axis = 0)
        reward = 0
        for steps in range(args.max_step):

            ac = agent.model(np.expand_dims(obs, axis = 0))
            n_obs, rew, done, _ = env.step(ac[0])
            states_to_be_labled = np.vstack([states_to_be_labled, np.expand_dims(n_obs, axis = 0)])
            reward += rew

            env.render()

            if done:
                break
            obs = n_obs

        print("Episode : {} \n Reward : {}".format(ep, reward))

        acs = expert_model(states_to_be_labled)

        rep_buffer.push(np.concatenate([states_to_be_labled, acs], axis = 1))














if __name__ == "__main__" :

    args = parser.parse_args()
    main(args)