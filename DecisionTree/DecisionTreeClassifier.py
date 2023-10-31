import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if (len(unique_classes) == 1) or (depth == self.max_depth) or (n_samples < self.min_samples_split):
            # If all samples have the same class or the tree depth limit is reached,
            # create a leaf node with the most common class
            return unique_classes[np.argmax(class_counts)]

        # Find the best split based on information gain
        best_split = self._find_best_split(X, y)

        if best_split is None:
            # If no split improves information gain, create a leaf node
            return unique_classes[np.argmax(class_counts)]

        # Create a decision node based on the best split
        feature_index, threshold, gini = best_split
        node = {}
        node["feature_index"] = feature_index
        node["threshold"] = threshold
        node["left"] = self._build_tree(X[X[:, feature_index] <= threshold], y[X[:, feature_index] <= threshold], depth + 1)
        node["right"] = self._build_tree(X[X[:, feature_index] > threshold], y[X[:, feature_index] > threshold], depth + 1)
        node["gini"] = gini
        node["num_samples"] = class_counts
        node["depth"] = depth
        return node

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape

        gini = self._calculate_gini(y) 

        best_info_gain = 0
        best_split = None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                y_left = y[X[:, feature_index] <= threshold]
                y_right = y[X[:, feature_index] > threshold]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                p_left = len(y_left) / n_samples
                p_right = len(y_right) / n_samples
                gain = gini - (p_left * self._calculate_gini(y_left) + p_right * self._calculate_gini(y_right))

                if gain > best_info_gain:
                    best_info_gain = gain
                    best_split = (feature_index, threshold, gini)

        return best_split

    def _calculate_gini(self, y):
        _, class_counts = np.unique(y, return_counts=True)
        return 1.0 - sum((count / len(y)) ** 2 for count in class_counts)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, np.int64):
            return tree
        feature_index, threshold, left_tree, right_tree = tree["feature_index"], tree["threshold"], tree["left"], tree["right"]
        if x[feature_index] <= threshold:
            return self._predict_tree(x, left_tree)
        else:
            return self._predict_tree(x, right_tree)

    

# Example usage:
if __name__ == '__main__':
    # Generate some example data
    #classifier 
    # Example usage:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier as skverdtc

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    clf2 = skverdtc()
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


