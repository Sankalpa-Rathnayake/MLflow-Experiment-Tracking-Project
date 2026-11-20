import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42

# Load data and make ONE fixed split used by every run
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 10 experiments: (run name, model, params to log)
experiments = [
    ("baseline_logreg", LogisticRegression(max_iter=1000, random_state=SEED),
        {"model": "LogisticRegression", "C": 1.0}),
    ("logreg_C0.1", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
        {"model": "LogisticRegression", "C": 0.1}),
    ("logreg_C10", LogisticRegression(C=10, max_iter=1000, random_state=SEED),
        {"model": "LogisticRegression", "C": 10}),
    ("rf_50", RandomForestClassifier(n_estimators=50, random_state=SEED),
        {"model": "RandomForest", "n_estimators": 50}),
    ("rf_200", RandomForestClassifier(n_estimators=200, random_state=SEED),
        {"model": "RandomForest", "n_estimators": 200}),
    ("rf_200_depth3", RandomForestClassifier(n_estimators=200, max_depth=3, random_state=SEED),
        {"model": "RandomForest", "n_estimators": 200, "max_depth": 3}),
    ("svm_linear", SVC(kernel="linear", probability=True, random_state=SEED),
        {"model": "SVM", "kernel": "linear"}),
    ("svm_rbf", SVC(kernel="rbf", probability=True, random_state=SEED),
        {"model": "SVM", "kernel": "rbf"}),
    ("svm_rbf_C10", SVC(kernel="rbf", C=10, probability=True, random_state=SEED),
        {"model": "SVM", "kernel": "rbf", "C": 10}),
    ("svm_rbf_C0.1", SVC(kernel="rbf", C=0.1, probability=True, random_state=SEED),
        {"model": "SVM", "kernel": "rbf", "C": 0.1}),
]

mlflow.set_experiment("breast_cancer_comparison")

for run_name, model, params in experiments:
    with mlflow.start_run(run_name=run_name):
        pipeline = make_pipeline(StandardScaler(), model)

        # 5-fold cross-validation on the training data only
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

        # Train on full training set, evaluate on held-out test set
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        # Log everything
        mlflow.log_params(params)
        mlflow.log_param("seed", SEED)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("test_f1", f1_score(y_test, preds))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, probs))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"{run_name}: CV F1 = {cv_scores.mean():.3f}, Test F1 = {f1_score(y_test, preds):.3f}")