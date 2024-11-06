import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def generate_training_dataset(df, feature_list, label="blur"):

    X = df[feature_list]  # Replace with your feature column names
    y = df['blur'] 
# Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    training_dict = {"x_train":X_train, "x_test":X_test, "y_train":y_train, "y_test":y_test}
    return training_dict


def train_model(training_dict):

        # Define the model with basic parameters
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    X_train = training_dict["x_train"]
    y_train = training_dict["y_train"]
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    return best_rf


def predict_test_set(training_dict, model):

    X_test = training_dict["x_test"]
    y_test = training_dict["y_test"]

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)


def blur_classifier(df, feature_list):

    training_dict = generate_training_dataset(df, feature_list)
    blur_model = train_model(training_dict)
    predict_test_set(training_dict, blur_model)
    return blur_model