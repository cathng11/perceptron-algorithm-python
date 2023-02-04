import numpy as np
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


if __name__ == "__main__":
    X = np.array([[10, 5, 7, 4, 4, 5, 8, 8, 7, 6], [
                 1, 2, 1, 2.5, 1, 0.5, 0.5, 1, 0.3, 0.3]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, 1, -1, -1])
    X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    X = X.transpose()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    perceptron = Perceptron(learning_rate=0.0001, n_iters=1000)
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(np.array([[1, 4, 3]]))
    print(predictions)
    print("Perceptron classification accuracy",
          perceptron.accuracy(y_test, predictions))
