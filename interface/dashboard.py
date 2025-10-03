from pathlib import Path
import joblib
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_FILENAME = BASE_PATH / "final-models" / "random-forest-model.pkl"
TEST_DATA_FILE = BASE_PATH / "datalake/data-for-model/test/test_sleep_cassette.parquet"

def prepare_X_y(df, features, target_col):
    X = pd.get_dummies(df, drop_first=True)
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    y = df[target_col]
    # transforma y em categórico, com ordem definida
    y = pd.Categorical(y, categories=sorted(y.unique()))
    return X, y

if __name__ == "__main__":
    # Carregar pacote
    explainer_package = joblib.load(MODEL_FILENAME)
    model = explainer_package["model"]
    features = explainer_package["features"]
    target = explainer_package["target"]

    # Dataset de teste
    df_test = pd.read_parquet(TEST_DATA_FILE)
    X_test, y_test = prepare_X_y(df_test, features, target)

    # Criar Explainer usando y categórico
    explainer = ClassifierExplainer(model, X_test, y_test)

    # Dashboard
    db = ExplainerDashboard(
        explainer,
        title="Dashboard Random Forest - Sleep Cassette",
        whatif=True,
        shap_interaction=True
    )
    db.run()