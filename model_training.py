import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255,
    validation_split=0.2,
    rotation_range=40, # Augmentation options
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_data.flow_from_directory(
    './dataset/',
    target_size=(300, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_data.flow_from_directory(
    './dataset/',
    target_size=(300, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training:")
history = model.fit(
    train_generator,
    #steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    #validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30,
    verbose=1
)

model.save('pattern_blue.keras')
print("Saved as pattern_blue.keras")