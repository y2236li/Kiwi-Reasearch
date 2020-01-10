import collections
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression



from env import environment
from mem import ReplayBuffer, build_dqn
from agent import Agent

'''Prepare Data'''
# data_path = "data/clean raw data.xlsx"
# df = pd.read_excel(data_path)
# X = df[df.columns[3:]]
# Y = df["Brix"].values
# X_matrix = X.to_numpy()
# nonzero_index = np.nonzero(X_matrix.sum(axis = 0))[0]
# X_matrix_nonzero = X_matrix[:, nonzero_index]
# X_nonzero_columns = X.columns[nonzero_index]

f_xy = open("pickles/XYtraintest.pk1", "rb")
X_train, X_test, Y_train, Y_test = pickle.load(f_xy)
f_xy.close()

f_a = open("pickles/accuracy_list.pk1", "rb")
accuracy_list = pickle.load(f_a)
f_a.close()


idol_range = 0
idol_steps = 5
detect_range = [-3, -1, 0, 1, 3] # also input dim
step_options = [-8, -3, -1, 0, 1, 3, 8] # also action space
xy = (X_train, X_test, Y_train, Y_test)

sample_com = 100
pls2 = PLSRegression(n_components=sample_com)
pls2.fit(X_train, Y_train)
Y_pred = pls2.predict(X_test)
score = r2_score(Y_pred, Y_test)
cur_pos = (sample_com, score)

mapp = (range(1, X_train.shape[1]), accuracy_list)
grid = range(1, X_train.shape[1])
env = environment(xy, cur_pos, mapp, grid, idol_range, idol_steps, detect_range, step_options)

lr = 0.1 # also alpha
n_games = 2000
agent = Agent(gamma=0.9, epsilon=1, alpha=lr, input_dims=len(detect_range) + 1,
              n_actions=len(step_options), action_space=step_options, mem_size=900, batch_size=64, epsilon_end=0.0)

# agent.load_model_()
scores = []
eps_history = []

all_history = collections.defaultdict(dict)
action_history_list = []

fig = plt.figure()
ax = fig.add_subplot(111)

for i in tqdm(range(n_games)):
    done: bool = False
    score: int = 0
    observation = env.reset(150)
    counter = 0
    while not done and counter <= 300:
        ax.clear()
        plt.title(f"game = {i}; step = {counter}")
        env.showCurPos()
        action = agent.choose_action(observation)
        action_history_list.append(action)
        observation_, reward, done = env.step(action)
        print(observation, action, env.st_history, reward)
        score += reward
        agent.remember(observation, agent.action_space.index(action), reward, observation_, int(done))
        observation = observation_

        agent.learn()

        all_history[i][counter] = env.cur_pos

        counter += 1

        scores.append(score)
    print(f"game {n_games} end with steps {counter}")