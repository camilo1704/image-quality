import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.models import save_model
import joblib
from src.utils.files import list_imgs_files, mkdir_p
from src.data_processing.compute_blur import apply_all_measures


def generate_training_dataset(df, feature_list, save_scaler_path, label="blur"):

    X = df[feature_list]  # Replace with your feature column names
    y = df['blur'] 
# Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, save_scaler_path)

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


def train_blur_classifier(df, feature_list, save_scaler_path, save_model_path):

    training_dict = generate_training_dataset(df, feature_list, save_scaler_path)
    blur_model = train_model(training_dict)
    predict_test_set(training_dict, blur_model)
    save_model(blur_model, save_model_path)
    return blur_model


def df_blur(file_names, labels):
    data = [[x,y] for x,y in zip(file_names, labels)]
    df_images = pd.DataFrame(columns=["img_name","blur"], data = data)
    df_images = apply_all_measures(df_images)
    return df_images

feature_list = ['gradient_energy', 'kurt', 'tef', "variance_of_laplacian", "blurriness_fft", "blurriness_wavelet", "contrast_operator", "blurriness_steerable"]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_path', type=str,
                        help='Path with all images')
    parser.add_argument('--save_artifacts_path', type=str,
                        help='Path to save model artifacts')

    args = parser.parse_args()
    imgs_path = args.images_path
    save_artifacts_path = args.save_artifacts_path

    mkdir_p(save_artifacts_path)
    save_scaler_path = os.path.join(save_artifacts_path, "scaler.pkl")
    save_model_path = os.path.join(save_artifacts_path, "blur_model.pkl")


    blur_imgs_files = list_imgs_files(os.path.join(imgs_path, "super_blur"))
    normal_imgs_files = list_imgs_files(os.path.join(imgs_path, "normal"))


    labels = [1]*len(blur_imgs_files) + [0]*len(normal_imgs_files)
    file_names = blur_imgs_files + normal_imgs_files
    df = df_blur(file_names, labels)

    train_blur_classifier(df,feature_list, save_scaler_path, save_model_path)
    