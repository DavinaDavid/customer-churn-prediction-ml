
from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
#
from sklearn.linear_model import *
from sklearn.ensemble import *

# =========================================================
# STEP 1 - LOAD DATASETS
# =========================================================

def load_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets.

    Parameters
    ----------
    dataset_path : str
        File path for training dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df : Training dataset
        test_df  : Test dataset
    """

    # TODO:
    # 1. Read dataset CSV file
    #train_df = pd.read_csv("dataset1_HW1.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Define training and test DataFrames
    # both datasets are defined above while reading csv

    print(train_df.head(5))
    print(test_df.head(5))
    
    # 3. Return both DataFrames
    return train_df, test_df



# =========================================================
# STEP 2 - DATASET OVERVIEW
# =========================================================

def dataset_overview(train_df: pd.DataFrame,test_df: pd.DataFrame) -> None:
    """
    Display basic dataset information.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    test_df : pd.DataFrame
        Test dataset.

    Returns
    -------
    None
    """

    # TODO:
    # 1. Print dataset shapes
    print(f"Train dataset Shape: {train_df.shape}")
    print(f"Test dataset Shape: {test_df.shape}")

    # 2. Display missing values
    print("Missing values in Train dataset:")
    print(train_df.isnull().sum())

    print("Missing values in Test dataset:")
    print(test_df.isnull().sum())

    # 3. Display basic statistics
    print("Train Dataset Statistics:")
    print(round(train_df.describe(),2))

    print("Test Dataset Statistics:")
    print(round(test_df.describe(),2))

# Call Function
#dataset_overview(train_df, test_df)
 


# =========================================================
# STEP 3 - DEFINE FEATURES AND TARGET
# =========================================================

def split_features_target(train_df: pd.DataFrame,test_df: pd.DataFrame,target_column: str) -> Tuple[
    pd.DataFrame,pd.Series,pd.DataFrame,pd.Series]:
    """
    Separate features and target variable.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    test_df : pd.DataFrame
        Test dataset.

    target_column : str
        Name of target column.

    Returns
    -------
    Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series
    ]
        X_train : Training features
        y_train : Training labels
        X_test  : Test features
        y_test  : Test labels
    """

    # TODO:
    # 1. Split features and labels
    X_train = train_df.drop(target_column, axis=1) # features (all columns except target column)
    y_train = train_df[target_column] # target variable

    X_test = test_df.drop(target_column, axis=1) 
    y_test = test_df[target_column] #targeted feature

    print("Training features (X_train): " , list(X_train.columns))
    print("Training target (y_train): ",  y_train.name)

    print("Test features (X_test): " , list(X_test.columns))
    print("Test target (y_test): ", y_test.name)


    # 2. Return X_train, y_train, X_test, y_test
    return X_train, y_train, X_test, y_test

#Call Function
#X_train, y_train, X_test, y_test = split_features_target(train_df, test_df, "churn")

#print("Training features (X_train): " , list(X_train.columns))
#print("Training target (y_train): ",  y_train.name)

#print("Test features (X_test): " , list(X_test.columns))
#print("Test target (y_test): ", y_test.name)


# =========================================================
# STEP 4 - IDENTIFY COLUMN TYPES
# =========================================================

def identify_column_types( X_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical columns.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataset.

    Returns
    -------
    Tuple[List[str], List[str]]
        categorical_cols : List of categorical columns
        numeric_cols     : List of numerical columns
    """

    # TODO:
    print(X_train.info())
    print(X_train["gender"].dtype)
    print(X_train["region"].dtype)

    #My X_train has dtypes: int64, float64, str
    # But Categorical columns are stored as 'object' or 'string' in pandas,
    # even if they appear as 'str' in the dataset display.
    # Therefore, it is better to include both 'object' and 'string' types for robustness.



    # 1. Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    # 2. Identify numerical columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numerical columns:", numeric_cols)


    # 3. Return both lists
    return categorical_cols, numeric_cols

# Call Function
#cat_cols, num_cols = identify_column_types(X_train)

#print("Categorical columns:", cat_cols)
#print("Numerical columns:", num_cols)


# =========================================================
# STEP 5 - BUILD PREPROCESSOR
# =========================================================

def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Create preprocessing pipeline.

    Parameters
    ----------
    numeric_cols : List[str]
        Numerical feature column names.

    categorical_cols : List[str]
        Categorical feature column names.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline.
    """

    # TODO:
    # 1. Create numerical pipeline
    numeric_pipeline = Pipeline([
        #filled missing values with MEAN
        ("imputer", SimpleImputer(strategy="mean")),
        #scaled values to same range (mean=0, std=1)
        ("scaler", StandardScaler())
    ])

    # 2. Create categorical pipeline
    categorical_pipeline = Pipeline([
        # filled missing values with MOST FREQUENT value
        ("imputer", SimpleImputer(strategy="most_frequent")),
        #converted categories into 0-1 columns
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    # 3. Combine using ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    # 4. Return preprocessor
    print(preprocessor)
    return preprocessor

#preprocessor = build_preprocessor(num_cols, cat_cols)

# =========================================================
# STEP 6 - BUILD MODEL PIPELINE
# =========================================================

def build_model(
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    Create machine learning pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Data preprocessing pipeline.

    Returns
    -------
    Pipeline
        Full machine learning pipeline.
    """

    # TODO:
    # 1. Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    # 2. Combine preprocessing + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("\n----------PIPELINE BUILDING---------")
    print("\n Step 1: Preprocessor (ColumnTransformer)")
    print(preprocessor)

    print("\n Step 2: Model (RandomForestClassifier)")
    print(model)

    print("\n Step 3: Final Combined Pipeline")
    print(pipeline)
    print("-------------------------------------\n")

    # 3. Return pipeline
    return pipeline

#pipeline = build_model(preprocessor)


# =========================================================
# STEP 7 - TRAIN MODEL
# =========================================================

def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train machine learning model.

    Parameters
    ----------
    model : Pipeline
        Machine learning pipeline.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training labels.

    Returns
    -------
    Pipeline
        Trained model.
    """

    # TODO:
    # 1. Fit model
    model.fit(X_train, y_train)

    # 2. Return trained model
    return model


#trained_model = train_model(pipeline, X_train, y_train)

#print(f"My Pipeline Structure: {trained_model}")


# =========================================================
# STEP 8 - MAKE PREDICTIONS
# =========================================================

def make_predictions(
    model: Pipeline,
    X_test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate predictions and probabilities.

    Parameters
    ----------
    model : Pipeline
        Trained machine learning model.

    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        y_pred : Predicted labels
        y_prob : Prediction probabilities
    """

    # TODO:
    # 1. Predict class labels
    #y_pred = model.predict(X_test)
    y_pred = np.array(model.predict(X_test))


    # 2. Predict probabilities
    #y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1
    y_prob = np.array(model.predict_proba(X_test)[:, 1])



    # 3. Return both outputs
    return pd.Series(y_pred), pd.Series(y_prob)

#Call Function
#y_pred, y_prob = make_predictions(trained_model, X_test)

#print("Predictions:")
#print(y_pred.head())

#print("\nProbabilities:")
#print(y_prob.head())

# =========================================================
# STEP 9 - EVALUATE MODEL
# =========================================================

def evaluate_model(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series
) -> None:
    """
    Evaluate classification performance.

    Parameters
    ----------
    y_test : pd.Series
        True labels.

    y_pred : pd.Series
        Predicted labels.

    y_prob : pd.Series
        Predicted probabilities.

    Returns
    -------
    None
    """

    # TODO:
    # 1. Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    

    # 2. Print model performance

    print("\n----MODEL EVALUATION------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

    print("---------------\n")

#Function call
#evaluate_model(y_test, y_pred, y_prob)


# =========================================================
# STEP 10 - CONFUSION MATRIX
# =========================================================

def display_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series
) -> None:
    """
    Display confusion matrix.

    Parameters
    ----------
    y_test : pd.Series
        True labels.

    y_pred : pd.Series
        Predicted labels.

    Returns
    -------
    None
    """

    # TODO:
    # 1. Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # 2. Display results

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\nCONFUSION MATRIX")
    print("-----------|------------------|-------------------")
    print("                Predicted 0      Predicted 1")
    print("-----------|------------------|-----------------")
    print(f"Actual 0            {tn}               {fp}")
    print(f"Actual 1            {fn}               {tp}")
    print("-----------|------------------|-----------------\n")


#display_confusion_matrix(y_test, y_pred) 

# =========================================================
#  ROC Curve
# =========================================================

def plot_roc_curve(y_test: pd.Series,y_prob: pd.Series) -> None:
  
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.show()

# =========================================================
#  PRECISION RECALL CURVE
# =========================================================

def plot_precision_recall_curve(y_test: pd.Series,y_prob: pd.Series) -> None:
    """
    Plot Precision-Recall curve.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure()
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.show()


# =========================================================
# MODEL COMPARISON
# =========================================================


def compare_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    
    ##----Compare multiple Classification Models

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "ROC AUC": round(roc_auc, 3)
        })

    results_dataframe = pd.DataFrame(results).round(3)

    print(results_dataframe)

    return results_dataframe

# =========================================================
# STEP 11 - FEATURE IMPORTANCE
# =========================================================

def feature_importance_analysis(
    model: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Analyze feature importance.

    Parameters
    ----------
    model : Pipeline
        Trained machine learning pipeline.

    numeric_cols : List[str]
        Numerical feature names.

    categorical_cols : List[str]
        Categorical feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importances.
    """

    # TODO:
    # 1. Extract feature names
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    # 2. Extract feature importance values
    trained_model = model.named_steps["model"]

    #feature_importances = trained_model.feature_importances_

    # Extract importance values

    # Tree-based models
    if hasattr(trained_model, "feature_importances_"):
        feature_importances = trained_model.feature_importances_

    # Logistic Regression
    elif hasattr(trained_model, "coef_"):
        feature_importances = np.abs(trained_model.coef_[0])

    else:
        raise ValueError("Unsupported model type")



    # 3. Create importance DataFrame
    feature_importance_dataframe = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances
    })
    # Round importance value
    feature_importance_dataframe["importance"] = feature_importance_dataframe["importance"].round(3)

    #Sort Importance Descendingly
    feature_importance_dataframe = feature_importance_dataframe.sort_values(
        by="importance",
        ascending=False
    )

    # 4. Plot top features
    top_features = feature_importance_dataframe.head(10)
    plt.figure()
    plt.bar(top_features["feature"], top_features["importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()

    # 5. Return DataFrame
    return feature_importance_dataframe

#Function call
#feature_importance_dataframe = feature_importance_analysis(
 #   trained_model,num_cols,cat_cols)

#print(feature_importance_dataframe.head(10))

# =========================================================
# STEP 12 - SAVE PREDICTIONS
# =========================================================

def save_predictions(
    test_df: pd.DataFrame,
    y_pred: pd.Series,
    y_prob: pd.Series,
    output_file: str
) -> None:
    """
    Save prediction results to CSV file.

    Parameters
    ----------
    test_df : pd.DataFrame
        Original test dataset.

    y_pred : pd.Series
        Predicted labels.

    y_prob : pd.Series
        Prediction probabilities.

    output_file : str
        Output CSV filename.

    Returns
    -------
    None
    """

    # TODO:
    # 1. Create output DataFrame
    ##copy in the new dataframe to avoid modifying in the original data
    output_dataframe = test_df.copy()
    
    # 2. Add predictions
    ##Inserting Prediction and Probabilities in the copied df
    output_dataframe["predicted_label"] = y_pred
    output_dataframe["predicted_probability"] = y_prob

    # 3. Save to CSV file
    ##Save prediction output as csv
    output_dataframe.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved successfully to: {output_file}\n")


if __name__ == "__main__":

    # TODO:
    #Use this main function to run and test your functions

    ## -------Code Execution for dataset1--------

     #train_df1, test_df1 = load_datasets("dataset1_HW1.csv") 
     train_df1, test_df1 = load_datasets("dataset1_HW1.csv" , "dataset1_HW1.csv")

     dataset_overview(train_df1, test_df1)

     X_train1, y_train1, X_test1, y_test1 = split_features_target(train_df1, test_df1, "churn")

     cat_cols1, num_cols1 = identify_column_types(X_train1)

     #feature_importance_dataframe = feature_importance_analysis(train_df1,num_cols1, cat_cols1)
     #print(feature_importance_dataframe.head(10))

    # --------------------------------
    # DATASET 1 - MULTI MODEL SECTION
    # ---------------------------------

    #for dataset1 using 3 models i will train model, make prediction, evalute results, save the files for each model and campare models

     train_models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
     }


     comparison_results1 = []

     for model_name, model in train_models.items():

      print(f"\n----------- {model_name}---------")

      pipeline1 = Pipeline([("preprocessor", build_preprocessor(num_cols1, cat_cols1)),("model", model)])

      pipeline1.fit(X_train1, y_train1)

     ##Make Prediction 
      y_pred1, y_prob1 = make_predictions(pipeline1, X_test1)

      print("Dataset1 mean prediction:", np.mean(y_pred1))  

      ##Train Model
      trained_model1 = train_model(pipeline1, X_train1, y_train1)

      ##Analysis Important features
      feature_importance_dataframe = feature_importance_analysis(trained_model1,num_cols1, cat_cols1)

      print(f"My Pipeline Structure: {trained_model1}")

    ##Evaluate Each model using confusion matrix, ROC curve, precision recall curve 
      evaluate_model(y_test1, y_pred1, y_prob1)

      display_confusion_matrix(y_test1, y_pred1)

      plot_roc_curve(y_test1, y_prob1)
      plot_precision_recall_curve(y_test1, y_prob1)


     ##Save each prediction for each model in a separate csv file
      save_predictions(test_df1, y_pred1, y_prob1,f"churn_dataset1_{model_name}.csv")

      comparison_results1.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test1, y_pred1),
        "F1": f1_score(y_test1, y_pred1),
        "ROC_AUC": roc_auc_score(y_test1, y_prob1)
      })

     comparison_df1 = pd.DataFrame(comparison_results1).round(3)
     print("\n----DATASET 1 MODEL COMPARISON---")
     print(comparison_df1)


     

      
    ## ---Code Execution for Dataset2----------
    
     train_df2, test_df2 = load_datasets("dataset2_HW1.csv","dataset2_HW1.csv")

     dataset_overview(train_df2, test_df2)

     X_train2, y_train2, X_test2, y_test2 = split_features_target(train_df2, test_df2, "churn")

     cat_cols2, num_cols2 = identify_column_types(X_train2)

     #feature_importance_dataframe = feature_importance_analysis(train_df1,num_cols1, cat_cols1)
     
     # --------------------------------------
     # DATASET 2 - MULTI MODEL SECTION
     # --------------------------------------

     #for dataset2 using 3 models i will predict, evaluate results and save the files and then campare models  

     test_models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
     }

     comparison_results2 = []

     for model_name, model in test_models.items():
    
      print(f"\n--------- {model_name} -----------")

      pipeline2 = Pipeline([
        ("preprocessor", build_preprocessor(num_cols2, cat_cols2)),
        ("model", model)
      ])

      pipeline2.fit(X_train2, y_train2)

      y_pred2, y_prob2 = make_predictions(pipeline2, X_test2)

      print("Dataset2 mean prediction:", np.mean(y_pred2))  

      trained_model2 = train_model(pipeline2, X_train2, y_train2)
      feature_importance_dataframe = feature_importance_analysis(trained_model2,num_cols2, cat_cols2)

      print(f"My Pipeline Structure: {trained_model2}")

      #y_pred2, y_prob2 = make_predictions(trained_model2, X_test2)


     ##Evaluate each model using confusion matrix, roc_curve and precision-recall curve
      evaluate_model(y_test2, y_pred2, y_prob2)

      display_confusion_matrix(y_test2, y_pred2)

      plot_roc_curve(y_test2, y_prob2)
      plot_precision_recall_curve(y_test2, y_prob2)

     #feature_importance_dataframe = feature_importance_analysis(train_df1,num_cols1, cat_cols1)

     ##Save predictions for each model in csv file
      save_predictions(test_df2, y_pred2, y_prob2, f"churn_dataset2_{model_name}.csv")

      comparison_results2.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test2, y_pred2),
        "F1": f1_score(y_test2, y_pred2),
        "ROC_AUC": roc_auc_score(y_test2, y_prob2)
      })

     ##Model Camparison for dataset2
     comparison_df2 = pd.DataFrame(comparison_results2).round(3)
     print("\n-----DATASET 2 MODEL COMPARISON-----")
     print(comparison_df2)
    




