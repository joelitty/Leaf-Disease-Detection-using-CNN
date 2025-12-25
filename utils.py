from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMAGE_SIZE, BATCH_SIZE, DATASET_DIR

def load_data():
    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        f"{DATASET_DIR}/train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_data = datagen.flow_from_directory(
        f"{DATASET_DIR}/val",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = datagen.flow_from_directory(
        f"{DATASET_DIR}/test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_data, val_data, test_data
