import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        mean_value = np.mean(y)

        # Stopping conditions
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return mean_value

        
        best_split = self._find_best_split(X, y)
        feature_index, threshold, best_mse = best_split

        if best_mse == float('inf'):
            return mean_value

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return mean_value

        node = {}
        node["feature_index"] = feature_index
        node["threshold"] = threshold
        node["left"] = self._build_tree(X[X[:, feature_index] <= threshold], y[X[:, feature_index] <= threshold], depth + 1)
        node["right"] = self._build_tree(X[X[:, feature_index] > threshold], y[X[:, feature_index] > threshold], depth + 1)
        node["mean"] = np.mean(y)
        node["num_samples"] = len(y)
        node["depth"] = depth
        return node

    
    def _find_best_split(self, X, y):
        # Find the best split
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse = self._calculate_mse(y_left, y_right)
                if mse < best_mse:
                    best_feature = feature
                    best_threshold = threshold
                    best_mse = mse
        return (best_feature, best_threshold, best_mse)

    def _calculate_mse(self, y1, y2):
        mse = np.mean((y1 - np.mean(y1)) ** 2) + np.mean((y2 - np.mean(y2)) ** 2)
        return mse

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, (int, float)):
            return tree
        feature, threshold, left_tree, right_tree = tree["feature_index"], tree["threshold"], tree["left"], tree["right"]
        if x[feature] <= threshold:
            return self._predict_tree(x, left_tree)
        else:
            return self._predict_tree(x, right_tree)
        


# Example usage:
if __name__ == '__main__':
    # Generate some example data
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor as skverdtr

    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # X = np.array([[1], [2], [3], [4], [5]])
    # y = np.array([1, 2, 3, 4, 5])

    # Create and fit the decision tree
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X_train, y_train)

    tree2 = skverdtr()
    tree2.fit(X_train, y_train)
    predictions2 = tree2.predict(X_test)

    # Make predictions
    # X_test = np.array([[2.5], [4.5]])
    predictions = tree.predict(X_test)
    print(predictions)
    print(mean_squared_error(y_test, predictions))

    print(mean_squared_error(y_test, predictions2))



