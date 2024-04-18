from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os


class Terrain:
    def __init__(self, filename):
        self.filename = filename
        
    def predictionterrain(self):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        
        TF_ENABLE_ONEDNN_OPTS=0

        # Load the model
        model_path = os.path.join("model", "keras_model.h5")
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Load the labels
        labels_path = os.path.join("model", "labels.txt")
        class_names = open(labels_path, "r").readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(self.filename).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Remove leading/trailing whitespaces
        confidence_score = prediction[0][index]
        
        # Print prediction and confidence score
        print("Confidence Score:", confidence_score)

        # Return prediction result
        return [{"Class:": class_name}]


