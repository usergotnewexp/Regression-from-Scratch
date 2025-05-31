## More efficient and one step ahead of primitive 

import random

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.theta = None  # Parameters (theta0, theta1, ..., thetan)
    
    def fit(self, X, y):
        # Add intercept term (x0 = 1) to each sample
        X_with_intercept = [[1.0] + list(sample) for sample in X]
        n_samples = len(X_with_intercept)
        n_features = len(X_with_intercept[0])
        
        # Initialize parameters randomly
        self.theta = [random.random() for _ in range(n_features)]
        
        # Gradient descent
        for _ in range(self.n_iters):
            predictions = self._predict_batch(X_with_intercept)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            
            # Compute gradients for each parameter
            gradients = []
            for j in range(n_features):
                gradient_j = sum(errors[i] * X_with_intercept[i][j] for i in range(n_samples))
                gradients.append(gradient_j)
            
            # Update parameters
            for j in range(n_features):
                self.theta[j] -= self.lr * (gradients[j] / n_samples)
    
    def _predict_batch(self, X):
        return [sum(self.theta[j] * x[j] for j in range(len(x))) for x in X]
    
    def predict(self, X):
        X_with_intercept = [[1.0] + list(sample) for sample in X]
        return self._predict_batch(X_with_intercept)

# Example usage
if __name__ == "__main__":
    # Sample dataset: y = 1 + 2*x
    X_train = [[1], [4], [7], [8]]
    y_train = [3, 5, 7, 9]
    
    # Train model
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    
    # Predict
    X_test = [[5], [6]]
    predictions = model.predict(X_test)
    print("Predictions:", predictions)  # Expected: ~[11, 13]
    print("Learned parameters:", model.theta)  # Expected: ~[1.0, 2.0]