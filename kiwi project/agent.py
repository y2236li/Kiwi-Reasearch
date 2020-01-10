import numpy as np
from mem import ReplayBuffer, build_dqn
from keras.models import load_model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, action_space, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        """
        :param state: old state
        :param action: action index in action space
        :param reward: rewards gain from old state to new state
        :param state_: new state
        :param done: signal for stop moving
        :return: None

        Remember decision and store into memories
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = self.action_space[np.argmax(actions)]

        return action

    def learn(self):
        """
        :return: None
        Learn from the last step
        """
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

            action_values = np.array(range(len(self.action_space)), dtype=np.int8)
            action_indices = np.dot(action, action_values)
            q_eval = self.q_eval.predict(state)  # find the current Q value

            q_next = self.q_eval.predict(new_state)  # find the future Q values at the next state

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done
            # change the current Q matrix by reward of the transition plus the max future reward

            _ = self.q_eval.fit(state, q_target, verbose=0)  # fit model

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
                                                              self.epsilon_min else self.epsilon_min
            # reduce epsilon

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model_(self):
        self.q_eval = load_model(self.model_file)

