import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42
RUN_ID = "469062a48b254bfc89fd116209165fb4"

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

df = pd.DataFrame(X_test, columns=data.feature_names)
df["actual"] = y_test
df["predicted"] = preds
df["confidence"] = probs

errors = df[df["actual"] != df["predicted"]]
print(f"Total errors: {len(errors)} out of {len(df)}")
print(errors[["actual", "predicted", "confidence",
              "mean radius", "mean texture", "mean concavity"]])