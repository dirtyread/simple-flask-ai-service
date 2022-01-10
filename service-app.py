from flask import Flask, request, jsonify, redirect, Response, make_response
from common.bo.ClassificationImageEngine import ClassificationImageEngine
import io
import os


app = Flask(__name__)

execution_path = os.getcwd()

model = os.path.join(execution_path, "common", "models", "squeezenet.h5")
labels = os.path.join(execution_path, "common", "models", "squeezenet_labels.txt")
input_size = 227

predictor = ClassificationImageEngine(h5_model_path=model, labels_txt_file=labels, input_image_size=input_size)
app.predictor = predictor


@app.route("/")
def index():
    return redirect("healthcheck")


@app.route("/healthcheck")
def healthcheck():
    return "Health Check OK"


@app.route("/imageClassification", methods=['POST'])
def image_classification():
    file = request.get_data(cache=False, as_text=False, parse_form_data=False)
    if len(file) > 1:
        io_bytes_file = io.BytesIO(file)
        object_label = app.predictor.classify_from_stram(image_stream=io_bytes_file)

    response_obj = {"classification": object_label}

    response = make_response(response_obj)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    response.headers['mimetype'] = 'application/json'
    response.status_code = 200

    return response


if __name__ == "__main__":
    app.run(port=80, host="0.0.0.0")
