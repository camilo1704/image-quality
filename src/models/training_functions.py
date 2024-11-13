from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from src.utils.models import save_model
import os
import sys
import joblib
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def generate_training_dataset(df, feature_list, save_scaler_path, classification_name):
    """
    Generate training and testing datasets with standardized features for classification.
    :param df: DataFrame containing features and target labels.
    :param feature_list: (list of str)List of feature column names to include in the model.
        Example: ['dominant_color', 'saturation', 'colorfulness'...]
    :param save_scaler_path: (str)Path to save the scaler object for future use.
    :param classification_name: (str) Column name for the target labels. Example is "soil_color"
    :returns A dictionary containing the training (features) and testing (labels) datasets:
    """
    x = df[feature_list]
    y = df[classification_name]  # Dynamically use the label column specified in the arguments

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the feature columns
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Save the scaler for later use
    joblib.dump(scaler, save_scaler_path)

    training_dict = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
    return training_dict


def train_model(training_dict):
    """
    Train a Random Forest classifier using GridSearchCV to optimize hyperparameters.
    :param training_dict: training dataset {"x_train": Features, "y_train": Labels}
    :returns RandomForestClassifier: The best model obtained after GridSearchCV.
    """
    # Define the model with basic parameters
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    x_train = training_dict["x_train"]
    y_train = training_dict["y_train"]

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    return best_rf


def predict_test_set(training_dict, model, target_names):
    """
    Evaluate a trained model on the test set and print performance metrics.
    :param training_dict: A dictionary containing "x_test" and "y_test".
    :param model: (sklearn model) A trained machine learning model.
    :param target_names: (list of str) List of class names for the classification report.
    :returns A dictionary containing accuracy and the classification report as a string.
    """
    x_test = training_dict["x_test"]
    y_test = training_dict["y_test"]

    # Generate predictions
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    return {"accuracy": accuracy, "classification_report": report}


def train_generic_classifier(df, feature_list, save_scaler_path, save_model_path,
                             classification_name, target_names):
    """
    Train a generic classifier, evaluate it, and save the model and scaler.
    :param df: DataFrame containing the dataset.
    :param feature_list: (list of str) List of feature column names to use for training.
    :param save_scaler_path: (str) Path to save the scaler object.
    :param save_model_path: (str) Path to save the trained model.
    :param classification_name: (str) Column name for the target labels. Example is "soil_color"
    :param target_names: (list of str) List of class names for the classification report.
    :returns The trained classifier (sklearn model)
    """
    training_dict = generate_training_dataset(df, feature_list, save_scaler_path,
                                              classification_name=classification_name)
    model = train_model(training_dict)
    predict_test_set(training_dict, model, target_names=target_names)
    save_model(model, save_model_path)
    return model


def prepare_dataframe(file_names, labels, classification_name, feature_function):
    """
    Prepare a DataFrame for classification by combining file names and labels, and optionally applying features.
    :param file_names: (list of str) List of file paths or identifiers.
    :param labels: List of labels corresponding to the file names.
    :param classification_name: (str) Column name for the target labels. Example is "blur", "soil_color"
    :param feature_function: Function to apply features to the DataFrame.
        Usually apply_all_measures with all the attribute score functions to generate features to train the classifier
    :returns A DataFrame with file names, labels, and optionally computed features.
    """
    data = [[x, y] for x, y in zip(file_names, labels)]
    dataframe = pd.DataFrame(columns=["file_name", classification_name], data=data)
    dataframe = feature_function(dataframe)
    return dataframe
