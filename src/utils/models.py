import joblib

def save_model(model, filename):
    """
    Save a trained scikit-learn model to a file.
    
    Parameters:
    model: scikit-learn model instance
        The model to be saved.
    filename: str
        The path to the file where the model will be saved.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Load a scikit-learn model from a file.
    
    Parameters:
    filename: str
        The path to the file from which to load the model.
        
    Returns:
    model: scikit-learn model instance
        The loaded model.
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
