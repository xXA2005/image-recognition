from keras.models import load_model
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import requests
from keras.preprocessing.image import img_to_array
import io
import time

app = Flask(__name__)
np.set_printoptions(suppress=True)
model = load_model('model.h5')


class_names = ["dog","other","tree"]



def predict(url):
    res = requests.get(url, stream=True)
    res.raise_for_status()
    imageb = res.content
    img = img_to_array(Image.open(io.BytesIO(imageb)).resize((128, 128)))
    img = np.expand_dims(img, axis=0)
    start = time.time()
    prediction = model.predict(img)
    print(f"predicted image in {time.time()-start}")
    return np.argmax(prediction, axis=1)[0]


@app.route('/image', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        url = data.get('url', '')
        q = data.get('q', '')

        if not url or not q:
            return jsonify({'error': 'wtf bad json'}), 400

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return jsonify({'error': 'bad image url'}), 400

        p = class_names[predict(url)]

        return "true" if p in q else "false"

    except Exception as e:
        return jsonify({'error': e}), 400


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=6969)
