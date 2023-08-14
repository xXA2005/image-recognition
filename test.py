import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Load the Keras model
model = load_model('model.h5')

# Load and preprocess an image from a URL for prediction
def preprocess_image_from_url(image_url):
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image_content = response.content
    image = img_to_array(Image.open(io.BytesIO(image_content)).resize((128, 128)))
    image = np.expand_dims(image, axis=0)
    return image

# Make predictions using the model
def predict_image_from_url(image_url):
    image = preprocess_image_from_url(image_url)
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]

# URL of the image you want to predict
image_url = 'https://cdn.discordapp.com/attachments/1140020919855751222/1140746477493305415/6f5b373e-8059-4441-808b-8dd4fa9eb94e.png'

# Predict the class
predicted_class = predict_image_from_url(image_url)
print(f"Predicted class: {predicted_class}")
