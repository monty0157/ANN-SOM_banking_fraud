
#IMPORT DATASET
import pandas as pd
import numpy as np

dataset = pd.read_csv('./Data/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#IMPORT SOM CLASS
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#VISUALISING SOM
from pylab import bone, plot, colorbar, show, pcolor

#clean canvas
bone()
#add colored som
pcolor(som.distance_map().T)
#add color legend
colorbar()

#distinct between approved and unapproved customers
marker = ['o', 's']
color = ['r', 'g']

for i, x in enumerate(X):
    winner = som.winner(x)
    plot(winner[0] + 0.5,
         winner[1] + 0.5,
         marker[y[i]],
         markeredgecolor = color[y[i]])
    
show()

#GETTING LIST OF FRAUDS
mapping = som.win_map(X)
m_1 = mapping[(1,4)]
m_2 = mapping[(8,8)]
frauds = np.concatenate((m_1, m_2), axis = 0)
frauds = sc.inverse_transform(frauds)