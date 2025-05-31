def load_data(filename):
    """
    Load dataset from CSV file
    Format: feature1,feature2,...,price
    """
    X = []
    y = []
    with open(filename, 'r') as f:
        # Skip header
        header = next(f)
        for line in f:
            values = line.strip().split(',')
            # Convert all except last column to floats
            features = [float(x) for x in values[:-1]]
            X.append(features)
            y.append(float(values[-1]))
    return X, y

def normalize_features(X):
    """
    Min-max normalization: scale features to [0, 1] range
    """
    if not X:
        return []
    
    # Transpose features to column format
    features_col = list(zip(*X))
    normalized = []
    
    for col in features_col:
        min_val = min(col)
        max_val = max(col)
        normalized_col = []
        for val in col:
            # Avoid division by zero
            if max_val == min_val:
                normalized_col.append(0.5)
            else:
                normalized_col.append((val - min_val) / (max_val - min_val))
        normalized.append(normalized_col)
    
    # Transpose back to row format
    return list(zip(*normalized))

def save_model(model, feature_mins, feature_maxs, filename):
    """
    Save model parameters and normalization ranges
    """
    with open(filename, 'w') as f:
        # Save theta parameters
        f.write(','.join(str(t) for t in model.theta) + '\n')
        # Save min values for each feature
        f.write(','.join(str(m) for m in feature_mins) + '\n')
        # Save max values for each feature
        f.write(','.join(str(m) for m in feature_maxs))

def load_model(filename):
    """
    Load model parameters and normalization ranges
    """
    with open(filename, 'r') as f:
        theta_line = f.readline().strip().split(',')
        mins_line = f.readline().strip().split(',')
        maxs_line = f.readline().strip().split(',')
    
    return {
        'theta': [float(t) for t in theta_line],
        'feature_mins': [float(m) for m in mins_line],
        'feature_maxs': [float(m) for m in maxs_line]
    }