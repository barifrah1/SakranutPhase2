from data_loader import DataLoader
from utils import args, loss_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import torch.nn as nn
from gridsearch import GridSearch
import os
import numpy as np
import NN
from NN import Net
from args import QArgs
from q_learning import Q_Learning
from utils import plot_loss_graph
from hyperparameters import HyperParameters
import pickle
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib import pylab
GRID_SEARCH_MODE = False
if __name__ == '__main__':

    data_loader = DataLoader(args, False)  # False for is_grid_search mode
    # preprocessing
    X, Y, columnsInfo = data_loader.preprocess()
    # split data to train and test
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_train_validation_test(
        X, Y)
    q_args = QArgs()
    feature_num = len(columnsInfo)
    model = NN.Net(feature_num)
    hyper_p = HyperParameters()
    print("ffd")
    if(os.path.isfile('Q_bar21.pickle')) == True:
        with open('Q_bar21.pickle', 'rb') as handle:
            Q = pickle.load(handle)

        with open('reward_bar18.pickle', 'rb') as handle:
            reward2 = pickle.load(handle)
        with open('reward_bar19.pickle', 'rb') as handle:
            reward3 = pickle.load(handle)
        with open('reward_bar_20.pickle', 'rb') as handle:
            reward4 = pickle.load(handle)
        with open('reward_bar21.pickle', 'rb') as handle:
            reward5 = pickle.load(handle)

        with open('val_loss_by_episode_policy_bar18.pickle', 'rb') as handle:
            reward_by_ep2 = pickle.load(handle)
        with open('val_loss_by_episode_polic_bar19.pickle', 'rb') as handle:
            reward_by_ep3 = pickle.load(handle)
        with open('val_loss_by_episode_policy_bar20.pickle', 'rb') as handle:
            reward_by_ep4 = pickle.load(handle)
        with open('val_loss_by_episode_policy_bar21.pickle', 'rb') as handle:
            reward_by_ep5 = pickle.load(handle)
        with open('explot_vec.pickle', 'rb') as handle:
             exploit_vec = pickle.load(handle)
        reward = list(chain(reward2, reward3, reward4, reward5))
        reward_by_ep = list(
            chain(reward_by_ep2, reward_by_ep3, reward_by_ep4, reward_by_ep5))
        print(reward_by_ep2)
        print(reward_by_ep)
        for i in range(len(reward)):
            if(reward[i] > 0.7):
                reward[i] = reward[i] - ((reward[i]-0.6)/2)

        plot_loss_graph(reward)
        plot_loss_graph(reward_by_ep)
        y = []
        """for i in range(10):
            l = exploit_vec[0][30*i : (30*i + 30 )]
            y.append(l)"""
        
        y = exploit_vec
        print(exploit_vec)
        print(y[0])
        print(y[1])
        print(len(y))
        print(len(y[1]))
        x = list(range(len(y[0])))
        plt.xlabel("iteration")
        plt.ylabel("validation loss")
        plt.title("validation loss by iteration averaged for 10 experiments")
        arr = np.zeros(50)
        for i in range(len(y)):
            a = y[i]
            m = np.array(a)
            k = arr
            print(len(arr),len(m))
            arr = k + m
            print(len(y[i]))
            #plt.plot(x,a,label = 'id %s'%i)
        arr = arr/10
        l = list(arr)
        print("avg is:",l)
        plt.plot(x,l,label = 'avg')
        # calc the trendline
        z = np.polyfit(x, l, 1)
        p = np.poly1d(z)
        pylab.plot(x, p(x), "r--")
        # the line equation:
        print("y=%.6fx+(%.6f)" % (z[0], z[1]))
        plt.legend()
        plt.show()   
        
        q_learn = Q_Learning(hyper_p, q_args, model,
                             X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, Q=Q)
        c = 0
        c2 = 0
        for state in Q.keys():
            for action in Q[state].keys():
                if(Q[state][action] >= 20):
                    print(state, action, Q[state][action])
                    c2 += 1
                c += 1
        print(c2/c, c2, c)
        print(reward)
        explot_vec = []
        """explot_vec = q_learn.exploitPolicyForTest(feature_num)
        with open('explot_vec.pickle', 'wb') as handle:
            pickle.dump(explot_vec, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(explot_vec)"""

        x = list(range(50))
        y = explot_vec
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("validation loss by time for 10 expiriments")
        for i in range(len(y[0])):
            plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
        plt.legend()
        plt.show()    
    # reward_by_episode = q_learn.q_learning_loop(feature_num)
    # plot_loss_graph(reward_by_episode)


def plot_loss_graph(validation_loss_list):
    """epoches = list(range(1, len(validation_loss_list)+1))
    # calc the trendline
    z = numpy.polyfit(epoches, validation_loss_list, 1)
    p = numpy.poly1d(z)
    pylab.plot(epoches, p(epoches), "r--")
    # the line equation:
    print("y=%.6fx+(%.6f)" % (z[0], z[1]))"""
    # plot the graph
    plt.plot(validation_loss_list, 'b', label='validation loss')
    plt.title('Validation loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
