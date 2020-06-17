from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time


class artificialNeuron:
    # hidden layers is formated so [Number of Neurons] where index is the hidden layer number
    def __init__(self, idx, hiddenLayers, epocas, learningRate, activation):
        self.id = idx
        self.hiddenLayers = hiddenLayers
        self.epocas = epocas
        self.learningRate = learningRate
        self.activationFunction = activation
        self.createImportNeuralNetwork()

    def createImportNeuralNetwork(self):
        if len(self.hiddenLayers) == 0:
            self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(
            ), max_iter=self.epocas, learning_rate_init=self.learningRate, activation=self.activationFunction)
        elif len(self.hiddenLayers) == 1:
            self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(
                self.hiddenLayers[0]), max_iter=self.epocas, learning_rate_init=self.learningRate, activation=self.activationFunction)
        elif len(self.hiddenLayers) == 2:
            self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(
                self.hiddenLayers[0], self.hiddenLayers[1]), max_iter=self.epocas, learning_rate_init=self.learningRate, activation=self.activationFunction)
        elif len(self.hiddenLayers) == 3:
            self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(
                self.hiddenLayers[0], self.hiddenLayers[1], self.hiddenLayers[2]), max_iter=self.epocas, learning_rate_init=self.learningRate, activation=self.activationFunction)

    def train(self, xData, yData):
        self.neuralNetwork.fit(xData, yData)

    # Using F1, because it seems the best metric for our case, the average between recall and precision
    def score(self, xDataTrain, yDataTrain):
        k = 10
        kfold = KFold(n_splits=k)
        scores = cross_val_score(
            self.neuralNetwork, xDataTrain, yDataTrain, cv=kfold, scoring='f1_macro')
        sumOf = 0
        for score in scores:
            sumOf += scores
        avg = sumOf/len(scores)
        print(avg)


class neuralTester:
    def __init__(self):
        self.id = 0
        self.loadDataset()
        self.networks = []
        self.scores = []
        self.scoresWithId = []

    def loadDataset(self):
        pima = pd.read_csv('dataSet.csv', sep=";")
        x = pima.iloc[:, 0:11]
        y = pima.iloc[:, 11]
        standarizacion = StandardScaler().fit_transform(x)
        xStandard = pd.DataFrame(data=standarizacion, columns=x.columns)
        xStandard.head()
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            xStandard, y, test_size=0.2)

    def createNeuralNetwork(self):
        self.startTime = time.time()
        epocas = [10, 100, 500, 1000]
        learningRates = [0.05, 0.1, 0.3]
        activations = ["identity", "logistic", "tanh", "relu"]
        # 0 Hidden Layers
        for epoca in range(len(epocas)-2):
            for learningRate in learningRates:
                for activation in activations:
                    aux = artificialNeuron(
                        self.id, [], epocas[epoca], learningRate, activation)
                    self.networks.append(aux)
                    self.id += 1

        # 1 Hidden Layers
        for first in range(1, 6):
            for epoca in range(1, len(epocas)-1):
                for learningRate in learningRates:
                    for activation in activations:
                        aux = artificialNeuron(
                            self.id, [first], epocas[epoca], learningRate, activation)
                        self.networks.append(aux)
                        self.id += 1

        # 2 Hidden Layers
        for first in range(1, 6):
            for second in range(1, 5):
                for epoca in range(2, len(epocas)):
                    for learningRate in learningRates:
                        for activation in activations:
                            aux = artificialNeuron(
                                self.id, [first, second], epocas[epoca], learningRate, activation)
                            self.networks.append(aux)
                            self.id += 1

        # 3 Hidden Layers
        for first in range(1, 6):
            for second in range(1, 5):
                for third in range(1, 4):
                    for i in range(2, len(epocas)):
                        for learningRate in learningRates:
                            for activation in activations:
                                aux = artificialNeuron(
                                    self.id, [first, second, third], epocas[i], learningRate, activation)
                                self.networks.append(aux)
                                self.id += 1

    def createNeuralNetwork2(self):
        self.startTime = time.time()
        epocas = [200, 500, 1000, 2000]
        learningRates = [0.05, 0.06, 0.07, 0.1]
        activations = ["identity", "logistic", "tanh", "relu"]
        # 0 Hidden Layers
        for epoca in range(len(epocas)):
            for learningRate in learningRates:
                for activation in activations:
                    aux = artificialNeuron(
                        self.id, [], epocas[epoca], learningRate, activation)
                    self.networks.append(aux)
                    self.id += 1

        # 1 Hidden Layers
        for first in range(1, 6):
            for epoca in range(len(epocas)):
                for learningRate in learningRates:
                    for activation in activations:
                        aux = artificialNeuron(
                            self.id, [first], epocas[epoca], learningRate, activation)
                        self.networks.append(aux)
                        self.id += 1

        # 2 Hidden Layers
        for first in range(1, 6):
            for second in range(1, 5):
                for epoca in range(len(epocas)):
                    for learningRate in learningRates:
                        for activation in activations:
                            aux = artificialNeuron(
                                self.id, [first, second], epocas[epoca], learningRate, activation)
                            self.networks.append(aux)
                            self.id += 1

        # 3 Hidden Layers
        for first in range(1, 6):
            for second in range(1, 5):
                for third in range(1, 5):
                    for i in range(len(epocas)):
                        for learningRate in learningRates:
                            for activation in activations:
                                aux = artificialNeuron(
                                    self.id, [first, second, third], epocas[i], learningRate, activation)
                                self.networks.append(aux)
                                self.id += 1

    def train(self):
        for i in range(len(self.networks)):
            print("Train")
            print(i)
            self.networks[i].train(self.xTrain, self.yTrain)

    # We are using F1 because we think it is the best metric for our case, getting the avg between recall and precision
    def writeScore(self, iteration):
        for i in range(len(self.networks)):
            print("Score")
            print(i)
            k = 10
            kfold = KFold(n_splits=k)
            score = cross_val_score(
                self.networks[i].neuralNetwork, self.xTrain, self.yTrain, cv=kfold, scoring='f1_macro')
            avg = np.mean(score)
            avg = round(avg, 3)
            self.scores.append(avg)

        if iteration == 1:
            textFile = open("scores.txt", "w")
            n = textFile.write(str(self.scores))
            textFile.close()
            textFile = open("executionTime.txt", "w")
            d = textFile.write(str(time.time()-self.startTime))
            textFile.close()
        else:
            textFile = open("scores2.txt", "w")
            n = textFile.write(str(self.scores))
            textFile.close()
            textFile = open("executionTime2.txt", "w")
            d = textFile.write(str(time.time()-self.startTime))
            textFile.close()

    # Read the score array from the txt. We saved it so we dont have to run the code every time, which takes about 1:30 hours
    def readScore(self):
        f = open("scores.txt", "r")
        lines = f.read().split(",")
        for i in range(len(lines)):
            if i == len(lines)-1:
                self.scores.append(float(lines[i][1:-1]))
            else:
                self.scores.append(float(lines[i][1:]))

    def findTop5Scores(self):
        for i in range(len(self.scores)):
            aux = [i, self.scores[i]]
            self.scoresWithId.append(aux)

        for i in range(len(self.scores)):
            for j in range(len(self.scores)-i-1):
                if self.scoresWithId[j][1] < self.scoresWithId[j+1][1]:
                    self.scoresWithId[j], self.scoresWithId[j +
                                                            1] = self.scoresWithId[j+1], self.scoresWithId[j]

        self.topScores = self.scoresWithId[:5]
        for i in range(len(self.topScores)):
            index = self.topScores[i][0]
            print("Neural Network number " + str(i) +
                  " -> value: " + str(self.topScores[i][1]), end=" ")
            print("id: " + str(index) + " Hidden Layers: " +
                  str(self.networks[index].hiddenLayers) + " Activation Func: " +
                  str(self.networks[index].activationFunction)
                  + " Learning Rate: " + str(self.networks[index].learningRate) + " Epocas: " + str(self.networks[index].epocas))

    def printNumberOfNets(self):
        print(self.id)


neuralTesterObj = neuralTester()
neuralTesterObj.createNeuralNetwork()
neuralTesterObj.readScore()
neuralTesterObj.findTop5Scores()
neuralTesterObj.printNumberOfNets()

#neuralTesterObj = neuralTester()
# neuralTesterObj.createNeuralNetwork2()
# neuralTesterObj.train()
# neuralTesterObj.writeScore(2)
# neuralTesterObj.printNumberOfNets()
