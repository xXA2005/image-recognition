from keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from keras.preprocessing.image import img_to_array


app = Flask(__name__)
np.set_printoptions(suppress=True)
model = load_model('model.keras')

class_names = ["car", "food", "other"]


def predict_image(image):
    img = image.resize((128, 128))
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return class_names[np.argmax(prediction, axis=1)[0]]


@app.route('/image', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        name = data.get('name', '')
        q = data.get('q', '')

        if not name or not q:
            return jsonify({'error': 'wtf bad json'}), 400

        image = Image.open(name)
        p = predict_image(image)

        return "true" if p in q else "false"

    except Exception as e:
        return jsonify({'error': e}), 400


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=6969)
