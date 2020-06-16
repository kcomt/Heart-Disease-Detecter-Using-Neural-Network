import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

pima = pd.read_csv('dataSet.csv', sep=';')
X = pima.iloc[:, 0:11]
Y = pima.iloc[:, 11]
estandarizacion = StandardScaler().fit_transform(X)
X_nuevo = pd.DataFrame(data=estandarizacion, columns=X.columns)
X_nuevo.head()
X_train, X_test, Y_train, Y_test = train_test_split(X_nuevo, Y, test_size=0.2)
mlp1 = MLPClassifier(hidden_layer_sizes=(2,), max_iter=5000,
                     learning_rate_init=0.001, activation='logistic')
mlp1.fit(X_train, Y_train)
print(X_train)
