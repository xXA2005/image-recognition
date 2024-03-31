from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import io
import time
import os

np.set_printoptions(suppress=True)
model = load_model('model.keras')


class_names = ['animal with four legs', 'other']


def predict(path):
    with open(path, "rb") as f:
        imageb = f.read()
    img = img_to_array(Image.open(io.BytesIO(imageb)))
    img = np.expand_dims(img, axis=0)

    try:
        prediction = model.predict(img)
        return class_names[np.argmax(prediction, axis=1)[0]]
    except Exception as e:
        print(e)
        return "error"


if __name__ == '__main__':
    start = time.time()
    l = len(os.listdir('testing'))
    for img in os.listdir('testing'):
        print(f'{img} is {predict(f"./testing/{img}")}')
    print(f"predicted {l} images in {time.time()-start}")
