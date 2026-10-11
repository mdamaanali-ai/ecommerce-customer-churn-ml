import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance

def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    return df

def find_target(df):
    candidates = ["Churn","churn","Exited","Customer_Status","Attrition","target","Target"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_target(y):
    if y.dtype == "object":
        vals = y.astype(str).str.strip().str.lower()
        mapping = {"yes":1,"y":1,"true":1,"1":1,"churned":1,"exited":1,"no":0,"n":0,"false":0,"0":0,"active":0}
        return vals.map(mapping).fillna(pd.factorize(vals)[0]).astype(int)
    return pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

def build_models(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=350, random_state=42,
                                                class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.08,
                                                            random_state=42)
    }
    return {name: Pipeline([("preprocess", pre), ("model", model)]) for name, model in models.items()}

def evaluate_models(df, target, threshold=.5):
    y = prepare_target(df[target])
    X = df.drop(columns=[target])
    if "Customer_ID" in X.columns:
        X = X.drop(columns=["Customer_ID"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )
    results, fitted = [], {}
    for name, pipe in build_models(X).items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
        pred = (proba >= threshold).astype(int)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test,pred),
            "Precision": precision_score(y_test,pred,zero_division=0),
            "Recall": recall_score(y_test,pred,zero_division=0),
            "F1": f1_score(y_test,pred,zero_division=0),
            "ROC-AUC": roc_auc_score(y_test,proba),
            "PR-AUC": average_precision_score(y_test,proba)
        })
        fitted[name] = (pipe, X_test, y_test, proba, pred)
    return pd.DataFrame(results).sort_values(["F1","ROC-AUC"], ascending=False), fitted

def permutation_importance_table(pipe, X_test, y_test):
    r = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42,
                                scoring="f1", n_jobs=-1)
    names = X_test.columns
    out = pd.DataFrame({"Feature":names, "Importance":r.importances_mean})
    return out.sort_values("Importance", ascending=False).head(15)
