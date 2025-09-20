import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model('pattern_blue.keras')

if len(sys.argv) < 2:
    print("Usage: python classify.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]

img = image.load_img(img_path, target_size=(300, 150))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

img_preprocessed = img_batch / 255.0

prediction = model.predict(img_preprocessed)

print("\nAnalyzing pattern...")
if prediction[0][0] > 0.5:
    confidence = prediction[0][0] * 100
    print(f"** !! PATTERN: BLUE !! **")
    print(f"Confidence: {confidence:.2f}%")
else:
    confidence = (1 - prediction[0][0]) * 100
    print("Pattern: Unknown. Not an Angel.")
    print(f"Confidence: {confidence:.2f}%")