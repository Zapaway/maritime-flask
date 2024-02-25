from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import io
from PIL import Image

app = Flask(__name__)
CORS(app)


@app.route('/capture/<path:file_url>', methods=["GET"])
def process_capture(file_url: str):
    file_url = file_url.replace("captures/", "captures%2F")
    entire_file_url = file_url + "?" + request.query_string.decode()
    capture_data = requests.get(entire_file_url).content

    capture_img = Image.open(io.BytesIO(capture_data))
    width, height = capture_img.size
    pixel_values = list(capture_img.getdata())

    print(width)
    print(height)
    print(pixel_values)

    return jsonify({"hello": "WOAHHHH"})


@app.route('/hello', methods=["GET"])
def hello():
    return jsonify({"hello": "YOOOO"})


# app.run(host="0.0.0.0", port=80)
