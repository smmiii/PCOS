from flask import Flask, request
from pcos_detection.test import predict_pcos

app = Flask(__name__)

@app.route("/", methods=['POST'])
def pcos_detection():
  image = request.files['image']
  image.save('temp_uploads/temp.png')

  model_path = './saved_model/pcos_detection_model.h5'
  result, confidence = predict_pcos('temp_uploads/temp.png', model_path)
  print(result, confidence)

  return { 'result': result, 'confidence': confidence }