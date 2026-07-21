"""Train and compare multiple classification models for student stress prediction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover
    lgb = None

try:
    import catboost as cb  # type: ignore
except Exception:  # pragma: no cover
    cb = None

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover
    shap = None

try:
    from src.preprocess import preprocess_dataset
    from src.utils import MODELS_DIR, get_model_artifact_paths, load_dataframe, save_dataframe, save_json, save_pickle
except ImportError:  # pragma: no cover - support direct execution
    from preprocess import preprocess_dataset
    from utils import MODELS_DIR, get_model_artifact_paths, load_dataframe, save_dataframe, save_json, save_pickle


def build_model_specs() -> list[dict[str, object]]:
    """Create a configuration list of models to evaluate."""
    specs = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0),
            "param_grid": {},
            "use_gridsearch": False,
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=3),
            "param_grid": {},
            "use_gridsearch": False,
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=42, n_estimators=20, max_depth=8, min_samples_leaf=2),
            "param_grid": {},
            "use_gridsearch": False,
        },
        {
            "name": "Naive Bayes",
            "estimator": GaussianNB(),
            "param_grid": {},
            "use_gridsearch": False,
        },
    ]
    return specs


def evaluate_model(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Evaluate a trained classifier and return metrics."""
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    try:
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_test)
        y_pred_bin = lb.transform(predictions)
        roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
    except ValueError:
        roc_auc = None
    return {"report": report, "confusion_matrix": cm.tolist(), "f1_score": round(float(f1), 4), "roc_auc": roc_auc}


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Save a confusion matrix figure."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names: list[str], model_name: str) -> None:
    """Save feature importance for tree-based and linear models."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        return
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis", hue="feature", dodge=False, legend=False)
    plt.title(f"Top Features - {model_name}")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.close()


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """Create a multiclass ROC AUC plot for the best model when possible."""
    if not hasattr(model, "predict_proba"):
        return
    probabilities = model.predict_proba(X_test)
    label_binarizer = LabelBinarizer()
    y_true_bin = label_binarizer.fit_transform(y_test)
    classes = label_binarizer.classes_
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plotted_any = False
    for class_index, class_name in enumerate(classes):
        try:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, thresholds = roc_curve(y_true_bin[:, class_index], probabilities[:, class_index])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")
            plotted_any = True
        except Exception:
            continue
    if not plotted_any:
        plt.close()
        return
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.close()


def explain_best_model(model, X_test: pd.DataFrame, model_name: str) -> None:
    """Attempt SHAP explainability for the best-performing model."""
    if shap is None:
        return
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_shap_summary.png")
        plt.close()
    except Exception:
        return


def train_models() -> None:
    """Train the configured models, save artifacts and summarize the results."""
    dataset_path = get_model_artifact_paths()["dataset"]
    data = load_dataframe(dataset_path)
    X_train, X_test, y_train, y_test = preprocess_dataset(dataset_path)

    specs = build_model_specs()
    results = []
    best_model = None
    best_name = ""
    best_score = -1.0

    for spec in specs:
        name = spec["name"]
        estimator = spec["estimator"]
        param_grid = spec.get("param_grid", {})
        if spec.get("use_gridsearch"):
            search = GridSearchCV(estimator, param_grid, cv=2, scoring="f1_weighted", n_jobs=1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            score = search.best_score_
        else:
            model = estimator
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)

        metrics = evaluate_model(name, model, X_test, y_test)
        results.append({"name": name, "score": round(float(score), 4), **metrics})
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), name)
        plot_feature_importance(model, X_train.columns.tolist(), name)
        plot_roc_curve(model, X_test, y_test, name)

        if float(metrics["f1_score"]) > best_score:
            best_score = float(metrics["f1_score"])
            best_model = model
            best_name = name

    save_json(results, MODELS_DIR / "model_metrics.json")
    save_pickle(best_model, MODELS_DIR / "best_model.pkl")
    save_dataframe(pd.DataFrame(results), MODELS_DIR / "model_comparison.csv")
    print(f"Best model: {best_name} with F1={best_score:.3f}")


if __name__ == "__main__":
    train_models()
