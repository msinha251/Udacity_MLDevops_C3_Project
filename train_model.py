# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

from ml.model import train_model, compute_model_metrics, inference, compute_slice_metrics
from ml.data import process_data
from ml.clean_data import basic_cleaning

# Load data
data = pd.read_csv("data/census.csv")

# Clean data
cleaned_data, cat_cols, num_cols = basic_cleaning(data, "data/census_cleaned.csv", "salary")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(cleaned_data, test_size=0.20)

# cat_features = [
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
# ]

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_cols, label="salary", training=True)

# Train and save a model.
model = train_model(X_train, y_train)

#predict on test data
X_test, y_test, _, _ = process_data(test, categorical_features=cat_cols, label="salary", training=False, encoder=encoder, lb=lb)
y_pred = inference(model, X_test)

# compute slice_feature metrics
slice_metrics = compute_slice_metrics(test, 'salary', cat_cols, 'education', model, encoder, lb)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {fbeta}")


#save model, encoder, and lb
joblib.dump(model, "model/model.joblib")
joblib.dump(encoder, "model/encoder.joblib")
joblib.dump(lb, "model/lb.joblib")


