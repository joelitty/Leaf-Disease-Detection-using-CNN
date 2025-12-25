import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import IMAGE_SIZE, MODEL_PATH, CLASS_NAMES

def predict_leaf(image_path):
    model = load_model(MODEL_PATH)

    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return CLASS_NAMES[class_index]

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    print("Prediction:", predict_leaf(img_path))
