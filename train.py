from linear_regression import LinearRegression
from utils import load_data, normalize_features, save_model

# Configuration
DATA_PATH = 'data/housing.csv'
MODEL_PATH = 'housing_model.txt'
LEARNING_RATE = 0.1
ITERATIONS = 5000

def main():
    print("Housing Price Prediction - Training")
    print("===================================")
    
    # 1. Load data
    print(f"Loading data from {DATA_PATH}...")
    X, y = load_data(DATA_PATH)
    print(f"Loaded {len(X)} samples with {len(X[0]) if X else 0} features")
    
    # 2. Get original feature ranges for later normalization
    if X:
        feature_mins = [min(col) for col in zip(*X)]
        feature_maxs = [max(col) for col in zip(*X)]
    else:
        feature_mins = []
        feature_maxs = []
    
    # 3. Normalize features
    print("Normalizing features...")
    X_normalized = normalize_features(X)
    
    # 4. Train model
    print(f"\nTraining model (lr={LEARNING_RATE}, iterations={ITERATIONS})...")
    model = LinearRegression(learning_rate=LEARNING_RATE, n_iters=ITERATIONS)
    model.fit(X_normalized, y)
    
    # 5. Save model
    save_model(model, feature_mins, feature_maxs, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # 6. Display results
    print("\nLearned parameters:")
    print(f"Intercept (θ₀): {model.theta[0]:.6f}")
    for i, coef in enumerate(model.theta[1:]):
        print(f"θ_{i+1} (Feature {i+1}): {coef:.6f}")
    
    # 7. Cost history visualization
    if model.costs:
        print("\nCost reduction during training:")
        for i, cost in enumerate(model.costs):
            if i % 5 == 0:  # Print every 5th cost
                print(f"Iteration {i*100}: Cost = {cost:.4f}")

if __name__ == "__main__":
    main()