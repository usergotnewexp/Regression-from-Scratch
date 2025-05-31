from utils import load_model

def predict_new_house(model_data, features):
    """
    Predict price for a new house using saved model
    """
    # Normalize features using original min/max
    normalized = []
    for i, val in enumerate(features):
        min_val = model_data['feature_mins'][i]
        max_val = model_data['feature_maxs'][i]
        if max_val == min_val:
            normalized.append(0.5)
        else:
            normalized.append((val - min_val) / (max_val - min_val))
    
    # Add intercept term (1.0)
    normalized_with_intercept = [1.0] + normalized
    
    # Calculate prediction: θ₀*1 + θ₁*x₁ + θ₂*x₂ + ...
    prediction = 0
    for j in range(len(model_data['theta'])):
        prediction += model_data['theta'][j] * normalized_with_intercept[j]
    
    return prediction

def main():
    print("Housing Price Prediction")
    print("========================")
    
    # 1. Load saved model
    try:
        model_data = load_model('housing_model.txt')
        print("Model loaded successfully")
        num_features = len(model_data['feature_mins'])
        print(f"Model expects {num_features} features")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Get input features from user
    features = []
    print("\nEnter house features:")
    for i in range(num_features):
        feature_val = float(input(f"  Feature {i+1}: "))
        features.append(feature_val)
    
    # 3. Make prediction
    try:
        price = predict_new_house(model_data, features)
        print(f"\nPredicted price: ${price:,.2f}")
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()