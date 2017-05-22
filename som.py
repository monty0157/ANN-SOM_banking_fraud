
#IMPORT DATASET
import pandas as pd

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