import numpy as np
from pmlb import fetch_data
import time

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion = 'sqrt'

    def fit(self, X, y, scores, k):
        self.tree = self._build_tree(X, y, scores, k, depth=0)

    def _build_tree(self, X, y, scores, k, depth):
        n_samples, n_features = X.shape
        mean_value = np.mean(y)

        # calculating scores for different classes for current leaf of this estimator
        residuals_sum = np.sum(y)
        score_prob = self._softmax(scores)
        eps = np.finfo(np.float32).eps
        score_prob = np.clip(score_prob, eps, 1 - eps)
        num_classes = len(scores[0])
        curr_score = residuals_sum / np.sum(score_prob[:, k] * (1-score_prob[:, k]))

        # Stopping conditions
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return curr_score

        
        best_split = self._find_best_split(X, y)
        feature_index, threshold, best_mse = best_split

        if best_mse == float('inf'):
            return curr_score

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return curr_score

        node = {}
        node["feature_index"] = feature_index
        node["threshold"] = threshold
        node["left"] = self._build_tree(X[X[:, feature_index] <= threshold], y[X[:, feature_index] <= threshold], scores[X[:, feature_index] <= threshold], k, depth + 1)
        node["right"] = self._build_tree(X[X[:, feature_index] > threshold], y[X[:, feature_index] > threshold], scores[X[:, feature_index] > threshold], k, depth + 1)
        node["mean"] = np.mean(y)
        node["num_samples"] = len(y)
        node["depth"] = depth
        return node

    
    def _find_best_split(self, X, y):
        # Find the best split
        _, n_f = X.shape
        n_features = np.sqrt(n_f)
        features_idx = np.random.choice(n_f, int(n_features), replace=False)
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature in features_idx:
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
    
    def _softmax(self, x):
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, (float)):
            return tree
        feature, threshold, left_tree, right_tree = tree["feature_index"], tree["threshold"], tree["left"], tree["right"]
        if x[feature] <= threshold:
            return self._predict_tree(x, left_tree)
        else:
            return self._predict_tree(x, right_tree)
        
class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators = []
        self.classes = None
        self.f_0 = None
        self.scores = []

    def fit(self, X, y):
        self.classes = np.arange(len(np.unique(y)))
        y_enc = self._encode_classes(y)

        # Initialize f_0(x)/scores as the logarithm of class probabilities
        
        # self.f_0 = np.log(class_probs / (1.0 - class_probs))
        # self.scores.append([self.f_0[(y_enc == 1)[i]][0] for i in range(len(y))])
        # temp = [self.f_0] * len(y)
        # self.scores = np.array([list(temp[i]) for i in range(len(temp))])
        self.scores = np.zeros(y_enc.shape)
        

        for m in range(self.n_estimators):
            class_trees = []
            for k in self.classes:

            # Compute negative gradient (pseudo-residuals)
                probabilities = self._softmax(self.scores)
                eps = np.finfo(np.float32).eps
                probabilities = np.clip(probabilities, eps, 1 - eps)
                pseudo_residuals = self._compute_pseudo_residuals(y_enc[:, k], probabilities[:, k])

                # Fit regression tree to pseudo-residuals
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, pseudo_residuals, self.scores, k)
                class_trees.append(tree)
                

                # Update f_m(x) using the new tree
                update = self.learning_rate * tree.predict(X)
                self.scores[:, k] = self.scores[:, k] + update
            self.estimators.append(class_trees)

    def _encode_classes(self, y):
        y_encoded = np.zeros((len(y), len(self.classes)))
        for i, cls in enumerate(self.classes):
            y_encoded[:, i] = (y == cls)
        return y_encoded

    def _compute_pseudo_residuals(self, y, f):
        residuals = (y - f)
        return residuals

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x)

    def predict_proba(self, X):
        # f_x = np.zeros((X.shape[0], len(self.classes)))
        f_x = np.zeros((X.shape[0], len(self.classes)))
        # f_x = [f_x] * X.shape[0]
        # f_x = np.array([list(f_x[i]) for i in range(len(f_x))])
        for k in self.classes:
            for class_tree in self.estimators:
                f_x[:, k] += self.learning_rate * class_tree[k].predict(X)
        return self._softmax(f_x)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
# Example usage:
if __name__ == '__main__':
    # Generate some example data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import GradientBoostingClassifier as GBC

    X, y = fetch_data('ann_thyroid', return_X_y=True, local_cache_dir='./')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and fit the decision tree
    
    tree = GradientBoostingClassifier(n_estimators=100)
    start_time = time.perf_counter()
    tree.fit(X_train, y_train)
    end_time = time.perf_counter()

    elapsed_time_microsec = (end_time - start_time) * 1000
    print(elapsed_time_microsec)
    

    # Make predictions
    predictions = tree.predict(X_test)
    print(predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Training Time:{elapsed_time_microsec:.2f} milliseconds")
    print()


    tree2 = GBC()
    start_time = time.perf_counter()
    tree2.fit(X_train, y_train)
    end_time = time.perf_counter()
    
    predictions = tree2.predict(X_test)
    print(predictions)
    elapsed_time_microsec = (end_time - start_time) * 1000
    

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print(f"Training Time:{elapsed_time_microsec:.2f} milliseconds")
    