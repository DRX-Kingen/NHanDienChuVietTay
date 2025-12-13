import os
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import load_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CSV_PATH = os.path.join(BASE_DIR, "A_Z Handwritten Data.csv")


X_train, X_test, y_train, y_test = load_dataset(CSV_PATH)


data_aug = models.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return x

inp = layers.Input(shape=(28, 28, 1))
x = data_aug(inp)

x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(26, activation="softmax")(x)

model = models.Model(inp, out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")


checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=12,   
    restore_best_weights=True
)


model.fit(
    X_train, y_train,
    epochs=60,            
    batch_size=128,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop]
)


loss, acc = model.evaluate(X_test, y_test)
print(f"Final test accuracy: {acc:.4f}")
