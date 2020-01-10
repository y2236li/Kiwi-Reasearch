import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape)) # input_shape is the number of state
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        """
        :param state: old state
        :param action: action index in action space
        :param reward: rewards gain from old state to new state
        :param state_: new state
        :param done: signal for stop moving
        :return: None

        save old state in self.state_memory
        save new state in self.new_state_memory
        save action in self.action_memory
        save reward in self.reward_memory
        save done in terminal memory
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions to action memory, if appropriate
        if self.discrete:
            actions = np.zeros((self.action_memory.shape[1]))
            actions[action] = 1
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        :param batch_size: size of batch
        :return: old states, actions, rewards, new states, terminal
        sample some transitions from memory
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    """
    :param lr: Learning rate
    :param n_actions: number of possible actions
    :param input_dims: input dimension
    :param fc1_dims: dimension of first layer fully connected hidden layer
    :param fc2_dims: dimension of second layer fully connected hiden layer
    :return: A neural network model for deep Q learning
    """
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model