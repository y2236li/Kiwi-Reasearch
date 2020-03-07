import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


class environment():
    def __init__(self, xy, cur_pos, mapp, grid, idol_range, idol_steps, detect_range, step_options=range(-10, 11)):
        # argument never change
        self.mapp = mapp  # map
        self.grid = grid  # all place where the climber could move
        self.idol_range = idol_range  # if the last idol_steps did not change the accuracy more than idol_range,
        # the policy is done
        self.idol_steps = idol_steps
        self.X_train, self.X_test, self.Y_train, self.Y_test = xy
        self.step_options = step_options
        self.detect_range = detect_range

        # argument that will change
        self.cur_pos = cur_pos  # current position
        self.st_history = []  # short term history of previous movement
        self.known_place_dict = {0: 0}  # record the accuracy got from every detection and movement

    def calAccuracy(self, x):
        pls2 = PLSRegression(n_components=x)
        pls2.fit(self.X_train, self.Y_train)
        Y_pred = pls2.predict(self.X_test)
        y = r2_score(Y_pred, self.Y_test)
        return y

    def observe(self, x, action, step):
        observation_ = []
        x_list = x + np.array(self.detect_range)

        for tmp_x in x_list:
            if tmp_x not in self.known_place_dict.keys():
                if tmp_x < self.grid[0] or tmp_x > self.grid[-1]: # when x is out of range
                    observation_.append(0)
                else:
                    tmp_y = self.calAccuracy(tmp_x)
                    self.known_place_dict[tmp_x] = tmp_y
                    observation_.append(tmp_y)
            else:
                observation_.append(self.known_place_dict[tmp_x])

        observation_ = np.array(observation_) - self.cur_pos[1]
        observation_ = np.append(observation_, x/40000)
        observation_ = np.append(observation_, step)
        return observation_

    def step(self, action, s):
        '''
        Return observation_, reward, done
        '''
        x = self.cur_pos[0]
        if x + action > self.grid[-1] or x + action < self.grid[0]:
            observation_ = np.zeros(len(self.detect_range)+1)
            observation_ = np.append(observation_, x/40000)
            return observation_, self.cur_pos[1], True

        x += action
        if x > self.grid[-1]:
            x = self.grid[-1]
        elif x < self.grid[0]:
            x = self.grid[0]

        if x in self.known_place_dict.keys():
            y = self.known_place_dict[x]
        else:
            y = self.calAccuracy(x)
            self.known_place_dict[x] = y
        self.cur_pos = (x, y)


        observation_ = self.observe(x, action, s)


        if len(self.st_history) >= self.idol_steps:
            self.st_history.pop(0)
        self.st_history.append(y)
        # reward = y
        # print("R1: reward = y")
        # reward = y + (1-y) * (2*y - max(self.st_history)-min(self.st_history))
        # print("R2: reward = y + (1-y) * max(self.st_history) -min(self.st_history)")
        alpha = 0.5
        reward = alpha*(y + (1-y) * (2*y - max(self.st_history)-min(self.st_history))) + \
                (1-alpha) * (1/(1+np.exp((s-150)/20)))
        # print("R3: alpha*(y + (1-y) * (2*y - max(self.st_history)-min(self.st_history))) + (1-alpha) * (1/(1+e**((s-150)/20)))")

        # print("***Short-Term History", self.st_history)
        # print("***Idle activity: ", max(self.st_history) - min(self.st_history))
        done = False
        if len(self.st_history) >= self.idol_steps:
            if max(self.st_history) - min(self.st_history) < self.idol_range:
                done = True

        return observation_, reward, done

    def reset(self, pos=None):
        if pos is None:
            x = np.random.choice(self.grid)
        else:
            x = pos
        self.cur_pos = (x, self.calAccuracy(x))
        self.st_history = []
        observation_ = self.observe(x, 0, 0)
        return observation_

    def showCurPos(self):
        plt.plot(*self.mapp, color="y")
        plt.scatter(*self.cur_pos, color="b")
        plt.draw()
        plt.pause(0.02)


if __name__ == "__main__":
    f_xy = open("pickles/XYtraintest.pk1", "rb")
    X_train, X_test, Y_train, Y_test = pickle.load(f_xy)
    f_xy.close()

    f_a = open("pickles/accuracy_list.pk1", "rb")
    accuracy_list = pickle.load(f_a)
    f_a.close()

    idol_range = 0
    idol_steps = 2
    detect_range = list(range(-1, 2))
    detect_range.remove(0)
    step_options = range(-2, 3)
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