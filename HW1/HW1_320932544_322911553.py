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
        self.ids = (320932544, 322911553)

    def distance(self, X1, X2):
        return np.power(sum(np.power(abs(X1-X2), self.p)), 1/self.p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.y_real = y
        self.y_pred = []
        self.X_train = X

    def calcDistances(self, X: np.ndarray, X_index: int) -> np.ndarray:
        distances = []
        for j in range(len(self.X_train)):
            dist = self.distance(X[X_index], self.X_train[j])
            distances.append([dist, j])
        return distances

    def getMajorityVote(self, distances: np.ndarray) -> np.ndarray:
        votes = []
        for distance, j in distances:
            votes.append(self.y_real[j])
        votes_dist = np.bincount(votes)
        if len(np.unique(votes_dist)) < len(votes_dist):
            closest_vote = distances[0]
            return self.y_real[closest_vote[1]]
        return votes_dist.argmax()


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        # find the distances
        for X_index in range(len(X)):
            distances = self.calcDistances(X, X_index)
            # sort the distances and take k nearest
            distances.sort()
            distances = distances[0:self.k]
            self.y_pred.append(self.getMajorityVote(distances))
        return self.y_pred

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
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
