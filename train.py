import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    iris = load_iris()
    return train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2%}")
    return accuracy


def save_model(model, path="iris_model.joblib"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()