from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time


class neuralNetwork:
    def __init__(self):
        self.loadDataset()
        self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=1000,
                                           learning_rate_init=0.05, activation='identity')

    def loadDataset(self):
        pima = pd.read_csv('dataSet.csv', sep=";")
        x = pima.iloc[:, 0:11]
        y = pima.iloc[:, 11]
        standarizacion = StandardScaler().fit_transform(x)
        xStandard = pd.DataFrame(data=standarizacion, columns=x.columns)
        xStandard.head()
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            xStandard, y, test_size=0.2)

    def train(self):
        self.neuralNetwork.fit(self.xTrain, self.yTrain)

    def predict(self, fixedAcidity, volatileAcidity, citricAcid, residualSugar, chlorides, freeSulfurDioxide, totalSulfurDioxide, density, pH, sulphates, alcohol):
        xPredict = np.array([fixedAcidity, volatileAcidity, citricAcid, residualSugar, chlorides,
                             freeSulfurDioxide, totalSulfurDioxide, density, pH, sulphates, alcohol])
        xPredict = xPredict.reshape((1, 11))
        quality = self.neuralNetwork.predict(xPredict)[0]
        return quality


neuralNetwork = neuralNetwork()
neuralNetwork.train()
print(neuralNetwork.predict(7.4, 0.7, 0, 1.9, 0.076, 11,
                            34, 0.9978, 3.51, 0.56, 9.4))
