
import argparse
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (123456789, 987654321)

    def distance(self, X1, X2):
        return np.linalg.norm(X1-X2)

    def train_test_split(self, X, y, test_size):
        # TODO: shuffle
        # np.random.shuffle(X)
        split_idx = int(len(X)*(1-test_size))
        x_train, x_test, y_train, y_test = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

        return [x_train, x_test, y_train, y_test]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        x_train, x_test, y_train, y_test = self.train_test_split(X, y, 0.2)

        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test

        # TODO - what should be here?

        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        y_pred = []
        # find the distances
        for i in range(len(self.X_test)):
            distances = []
            votes = []
            for j in range(len(self.X_train)):
                dist = self.distance(self.X_test[i], self.X_train[j])
                distances.append([dist, j])

            # sort the distances and take k nearest
            distances.sort()
            distances = distances[0:self.k]

            # get the corresponding votes
            for distance, j in distances:
                votes.append(self.y_train[j])
            print(votes)
            print(distances)

            ans = np.bincount(votes).argmax()
            y_pred.append(ans)

        # choose most common label


        # TODO - your code here
        pass

        ### Example code - don't use this:
        return y_pred


def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    # TODO: calc accuracy against y_test
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
