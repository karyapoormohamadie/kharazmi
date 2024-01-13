from fastapi import FastAPI
import tensorflow as tf
from PIL import Image,ImageFilter
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io

num_classes = ['A', 'B', 'C']
model = tf.keras.saving.load_model('model_wight_detector')

def predictor(img):
    
    img = tf.keras.utils.load_img(img)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    prediction = model.predict(img_array)

    score = tf.nn.softmax(prediction[0])
    predict = num_classes[np.argmax(score)], 100 * np.max(score)

    result = ''
    if predict[0] == 'A':
        result = 'skinny'
    elif predict[0] == 'B':
        result = 'fat'
    else:
        result = 'vulky'
    return result


app = FastAPI()

@app.get('/karya')
def wllcome():
    return'wellcome'


@app.post("/karya/faz1")
async def upload_photo(file: UploadFile = File(...)):
    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).resize((400,400))
    img.filter(ImageFilter.EDGE_ENHANCE()).save('img.jpg')

    result = predictor('img.jpg')

    return {result}