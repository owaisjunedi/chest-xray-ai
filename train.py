import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def train():
    print("Loading data...")
    # 1. Setup Generators
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
    train_gen = train_datagen.flow_from_directory(
        'data/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )

    # 2. Build Model (MobileNetV2 Transfer Learning)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 3. Train
    print("Training model...")
    model.fit(train_gen, epochs=5)

    # 4. Save the final result
    model.save('pneumonia_model.h5')
    print("Model saved to pneumonia_model.h5")

if __name__ == "__main__":
    train()