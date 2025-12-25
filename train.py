from model import build_cnn_model
from utils import load_data
from config import IMAGE_SIZE, EPOCHS, MODEL_PATH, CLASS_NAMES
import os

os.makedirs("saved_model", exist_ok=True)

train_data, val_data, _ = load_data()

model = build_cnn_model(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    num_classes=len(CLASS_NAMES)
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save(MODEL_PATH)
print("✅ Model trained and saved successfully")
