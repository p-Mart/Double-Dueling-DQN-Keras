from __future__ import division

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

#https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py
from gridworld import gameEnv

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import _Merge, Multiply

env_size = 5
env = gameEnv(partial=False, size=env_size)

class Experience():

    def __init__(self, buffer_size):
        
        self.replay_buffer = []
        self.buffer_size = buffer_size

    def storeExperience(self, exp):

        if(len(exp)+self.buffer_size >= len(self.replay_buffer)):
            del self.replay_buffer[:(len(exp)+len(self.replay_buffer) - self.buffer_size)]
        
        self.replay_buffer.extend(exp)

        return self.replay_buffer

    def sample(self, sample_size):
        return np.reshape(np.array(random.sample(self.replay_buffer, size)), (sample_size, env_size))

class QLayer(_Merge):
    '''Q Layer that merges an advantage and value layer'''
    def _merge_function(self, inputs):
        '''Assume that the inputs come in as [value, advantage]'''
        output = inputs[0] + (inputs[1] - K.mean(inputs[1], axis=1, keepdims=True))
        return output

class QNetwork():

    def __init__(self, h_size):
        self.inputs = Input(shape=(84,84,3))
        self.actions = Input(shape=(None,1), dtype='int32')
        self.actions_onehot = Lambda(K.one_hot, 
                                                            arguments={'num_classes':env.actions}, 
                                                            output_shape=(None, env.actions)
                                                          )(self.actions)

        x = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], input_shape=(-1, 84, 84, 3))(self.inputs)
        x = Conv2D(filters=64, kernel_size=[4,4],strides=[2,2])(x)
        x = Conv2D(filters=64, kernel_size=[3,3],strides=[1,1])(x)
        x = Conv2D(filters=h_size, kernel_size=[7,7],strides=[1,1])(x)

        #Splice outputs of last conv layer using lambda layer
        x_value = Lambda(lambda x: x[:,:,:h_size//2], output_shape=(h_size//2,))(x)
        x_advantage = Lambda(lambda x: x[:,:,h_size//2:], output_shape=(h_size//2,))(x)

        #Process spliced data stream into value and advantage function
        value = Dense(env.actions, input_shape=(h_size // 2, ), activation="linear")(x_value)
        advantage = Dense(env.actions, input_shape=(h_size // 2, ), activation="linear")(x_advantage)

        #Recombine value and advantage layers into Q layer
        q = QLayer()([value, advantage])

        self.q_out = Multiply()([q, self.actions_onehot])

        #need to figure out how to represent actions within training
        self.model = Model(inputs=[self.inputs, self.actions], outputs=self.q_out)
        self.model.compile(optimizer="Adam", loss="mean_squared_error")

        self.model.summary()


def resizeFrames(states):
    return np.reshape(states, [84*84*3])


h_size = 512
batch_size = 32
update_freq = 4
gamma = 0.9
start_eps = 1.
end_eps = 0.1
annealing_steps = 10000.
num_episodes = 10000
pre_train_steps = 10000
max_episode_length = 50
target_update_rate = 0.001

eps = start_eps
step_drop = (start_eps - end_eps) / annealing_steps

#store rewards and steps per episode
j_list = []
r_list = []
total_steps = 0

actor_network = QNetwork(h_size)
target_network = QNetwork(h_size)

experience = Experience(buffer_size=50000)

## Do this to periodically update the target network with ##
## the weights of the actor network                                   ##
#target_network.set_weights(actor_network.get_weights())

'''
for i in xrange(num_episodes):
    episode_exp = Experience(buffer_size=50000)
    s = env.reset()
    s = resizeFrames(s)
    done = False
    total_reward = 0
    j = 0

    while j < max_episode_length:
        j += 1

        if np.random.rand(1) < e or total_steps < pre_train_steps:
            a = np.random.randint(0, 4)
        else:
            prediction = actor_network.predict(s)
            a = np.argmax(prediction)

        s1, r, done = env.step(a)
        s1 = resizeFrames(s1)
        total_steps += 1
        episode_exp.storeExperience(np.reshape(np.array([s,a,r,s1,done]), [1, 5]))

        if total_steps > pre_train_steps:
            if eps > end_eps:
                eps -= step_drop

            if total_steps % update_freq == 0:
'''