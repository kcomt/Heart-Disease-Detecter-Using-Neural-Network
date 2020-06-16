from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


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
        epocas = [10, 100, 200, 500, 1000]
        learningRates = [0.001, 0.005, 0.01, 0.05,
                         0.07, 0.1, 0.2, 0.3, 0.5]
        activations = ["identity", "logistic", "tanh", "relu"]

        # 1 Hidden Layers
        for first in range(1, 9):
            for epoca in epocas:
                for learningRate in learningRates:
                    for activation in activations:
                        print(self.id)
                        aux = artificialNeuron(
                            self.id, [first], epocas, learningRate, activation)
                        aux.train(self.xTrain, self.yTrain)
                        self.networks.append(aux)
                        self.id += 1

        # 2 Hidden Layers
        for first in range(1, 9):
            for second in range(1, 9):
                for epoca in epocas:
                    for learningRate in learningRates:
                        for activation in activations:
                            aux = artificialNeuron(
                                self.id, [first, second], epocas, learningRate, activation)
                            self.networks.append(aux)
                            self.id += 1

        # 3 Hidden Layers
        for first in range(1, 9):
            for second in range(1, 9):
                for third in range(1, 9):
                    for i in range(3, len(epocas)):
                        for learningRate in learningRates:
                            for activation in activations:
                                aux = artificialNeuron(
                                    self.id, [first, second, third], epocas[i], learningRate, activation)
                                self.networks.append(aux)
                                self.id += 1

        # ai = artificialNeuron(0, [1, 2], 100, 0.3, "logistic")
        #ai.train(self.xTrain, self.yTrain)
        print(self.id)


neuralTesterObj = neuralTester()
neuralTesterObj.createNeuralNetwork()
