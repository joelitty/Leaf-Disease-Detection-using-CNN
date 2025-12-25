from tensorflow.keras.models import load_model
from utils import load_data
from config import MODEL_PATH

_, _, test_data = load_data()

model = load_model(MODEL_PATH)
loss, accuracy = model.evaluate(test_data)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
