"""Train EcoSort AI using the public TrashNet image dataset.

Expected dataset layout:
data/dataset-resized/cardboard/*.jpg
data/dataset-resized/glass/*.jpg
data/dataset-resized/metal/*.jpg
data/dataset-resized/paper/*.jpg
data/dataset-resized/plastic/*.jpg
data/dataset-resized/trash/*.jpg

The dataset is not bundled because it is a third-party dataset. See README.md for setup.
"""
from pathlib import Path
import json
import tensorflow as tf

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "dataset-resized"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    raise SystemExit("Dataset folder not found. Follow README.md to download and extract the dataset into data/dataset-resized/.")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
(MODEL_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_DIR / "ecosort_mobilenetv2.keras",
                                       monitor="val_accuracy", save_best_only=True)
]
model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
print("Saved model to models/ecosort_mobilenetv2.keras")
