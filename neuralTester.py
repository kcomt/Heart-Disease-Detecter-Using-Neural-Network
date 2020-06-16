from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import io
import pandas as pd


class artificialNeuron:
    # hidden layers is formated so [Number of Neurons] where index is the hidden layer number
    def __init__(self, id, hiddenLayers, epocas, learningRate, activation):
        self.id = id
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
        elif len(self.hiddenLayers) == 4:
            self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(
                self.hiddenLayers[0], self.hiddenLayers[1], self.hiddenLayers[2], self.hiddenLayers[3]), max_iter=self.epocas, learning_rate_init=self.learningRate, activation=self.activationFunction)
        elif len(self.hiddenLayers) == 5:
            self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(
                self.hiddenLayers[0], self.hiddenLayers[1], self.hiddenLayers[2], self.hiddenLayers[3], self.hiddenLayers[4]), max_iter=self.epocas, learning_rate_init=self.learningRate, activation=self.activationFunction)

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

    def loadDataset(self):
        pima = pd.read_csv('dataSet.csv', sep=' ')
        x = pima.iloc[:, 0:8]
        y = pima.iloc[:, 8]
        standarizacion = StandardScaler().fit_transform(x)
        xStandard = pd.DataFrame(data=standarizacion, columns=x.columns)
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            xStandard, y, test_size=0.2)

    # def createNeuralNetworks(self):


neuralTesterObj = neuralTester()
