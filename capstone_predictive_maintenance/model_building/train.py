import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

LOCAL_MLRUNS_PATH = os.path.join(os.getcwd(), "mlruns")
os.makedirs(LOCAL_MLRUNS_PATH, exist_ok=True)

# Set the URI to this specific subfolder
mlflow.set_tracking_uri(f"file://{LOCAL_MLRUNS_PATH}")
mlflow.set_experiment("Predictive_Maintenance_XGB")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

api = HfApi()

# Ensure login is performed before downloading
login(token=os.getenv("Predictive_Maintenance"))

repo_id = "sp1505/Predictive-Maintenance-Dataset"

# Download the files using hf_hub_download
Xtrain_path_local = hf_hub_download(
    repo_id=repo_id,
    filename="Xtrain.csv",
    repo_type="dataset"
)
Xtest_path_local = hf_hub_download(
    repo_id=repo_id,
    filename="Xtest.csv",
    repo_type="dataset"
)

ytrain_path_local = hf_hub_download(
    repo_id=repo_id,
    filename="ytrain.csv",
    repo_type="dataset"
)
ytest_path_local = hf_hub_download(
    repo_id=repo_id,
    filename="ytest.csv",
    repo_type="dataset"
)

# Load into pandas from local paths
Xtrain = pd.read_csv(Xtrain_path_local)
Xtest = pd.read_csv(Xtest_path_local)
ytrain = pd.read_csv(ytrain_path_local)
ytest = pd.read_csv(ytest_path_local)


# One-hot encode 'Type' and scale numeric features
# Check if 'Type' column exists in Xtrain. This was an issue in previous cells.
# Assuming 'Type' column is handled, but if not, this will cause an error.
# Based on the EDA, it looks like 'Type' was removed or not present in the final dataset used for X_train/X_test.
# The problem description mentions 'Engine_Condition' as categorical, not 'Type'.
# Let's adjust this based on the available data from df_capped which was used to create X_train and X_test
numeric_features = Xtrain.select_dtypes(include=['float64','int64']).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()

# If ytrain is a DataFrame, convert it to a Series for value_counts
if isinstance(ytrain, pd.DataFrame):
    ytrain_series = ytrain.squeeze() # Squeeze to Series if it's a single-column DataFrame
else:
    ytrain_series = ytrain

# Set the class weight to handle class imbalance
# Ensure to handle the case where a class might not exist, though unlikely in a split
class_0_count = ytrain_series.value_counts().get(0, 0)
class_1_count = ytrain_series.value_counts().get(1, 0)

if class_1_count > 0:
    class_weight = class_0_count / class_1_count
else:
    class_weight = 1.0 # Or handle as an error if class 1 must be present

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
) # Adjusted to use Xtrain columns directly

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_cv_recall", results['mean_test_score'][i])
            mlflow.log_metric("std_cv_recall", results['std_test_score'][i])

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.2
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_predictive_maintenance_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    model_repo_id = "sp1505/Predictive-Maintenace-Model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Space '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{model_repo_id}' not found. Creating new space...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
        print(f"Space '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_predictive_maintenance_model_v1.joblib",
        path_in_repo="best_predictive_maintenace_model_v1.joblib",
        repo_id=model_repo_id,
        repo_type=repo_type,
    )
