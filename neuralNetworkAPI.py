class neuralNetwork:
    def __init__(self):
        self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(2,), max_iter=5000,
                                           learning_rate_init=0.001, activation='logistic')
