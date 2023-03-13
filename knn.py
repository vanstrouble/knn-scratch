import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer


class KNN():
    def __init__(self, X_train, y_train, k=3) -> None:
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x):
        distances = [np.linalg.norm(np.array(x) - np.array(x_train)) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        return Counter(k_labels).most_common()[0][0]

    def get_predictions(self, X_test):
        return [self.predict(x) for x in X_test]


def evaluate(predictions, y_test):
    accuracy = np.sum(predictions == y_test) / len(y_test)
    # for index, element in enumerate(predictions):
    #     print(f'{X_test[index]}, pred: {element}, expected: {y_test[index]}')
    return np.around(accuracy, 2)


if __name__ == '__main__':
    # IrisPlant
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # plt.figure()
    # plt.scatter(X[:, 2], X[:, 3], c=y, s=20)
    # plt.show()

    # Breast Cancer
    # cancer = load_breast_cancer()
    # X, y = cancer.data, cancer.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Normal KNN without multiprocessing
    start = time.time()
    normal_cls = KNN(X_train, y_train, k=5)
    normal_pred = normal_cls.get_predictions(X_test)
    end = time.time()
    print('\nNormal KNN without multiprocessing')
    print(f'Accuracy: {evaluate(normal_pred, y_test)}, time: {end - start}')
    
    # Multiprocessing KNN
    # start = time.time()
    # mp_cls = KNN(X_train, y_train, k=5)
    # mp_pred = mp_cls.mp_get_predictions(X_test)
    # end = time.time()
    # print('\nMultiprocessing KNN')
    # print(f'Accuracy: {evaluate(mp_pred, y_test)}, time: {end - start}')
