import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("model_train.pkl")

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Inference
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Inference Accuracy: {accuracy:.4f}")
