from sklearn.metrics import accuracy_score


def evaluate_model(model, x_test, y_test) -> float:
    predictions = model.predict(x_test)
    return float(accuracy_score(y_test, predictions))
