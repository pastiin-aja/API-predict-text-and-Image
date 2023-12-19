import os
import tensorflow as tf
import pickle
import requests
from pydantic import BaseModel
from fastapi import FastAPI, Response, UploadFile,  HTTPException
from easyocr import Reader

# Load the CountVectorizer from the joblib file
loaded_vectorizer = pickle.load(open('./vectorizer.pickle', 'rb'))

# Initialize Model
# If you already put your model in the same folder as this main.py
# You can load .h5 model or any model below this line
model = tf.keras.models.load_model('./model_fraud_v1.h5')

class InputTextData(BaseModel):
    text: str

class InputImageData(BaseModel):
    url: str

class PredictionResult(BaseModel):
    text: str
    prediction: float 

def model_predict(text_vectorized):
    # Make predictions using the loaded model
    predictions = model.predict(text_vectorized)

    return predictions


app = FastAPI()

@app.get("/")
def index():
    return {"msg": "mainpage"}

@app.post("/text_predict")
def predictor(data: InputTextData):
    # Vectorize the input text
    text_vectorized = loaded_vectorizer.transform([data.text])

    result = model_predict(text_vectorized)
    return {"msg": result.tolist()}  # Convert predictions to a list

@app.post("/image_predict", response_model=PredictionResult)
async def extract_and_predict(data: InputImageData):
    try:
        response = requests.get(data.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

        # Checking if it's an image
        # content_type = response.headers.get("content-type", "")
        # if content_type not in ["image/jpeg", "image/png"]:
        #     return Response(content="File is Not an Image", status_code=400)

        # Use EasyOCR to extract text from the image
        reader = Reader(['en', 'id'])  # You can add more languages if needed
        image_bytes = response.content
        result = reader.readtext(image_bytes)

        # Extract text from the result
        text = " ".join([item[1] for item in result])

        # Vectorize the extracted text
        text_vectorized = loaded_vectorizer.transform([text])

        # Make predictions using the loaded model
        prediction = model_predict(text_vectorized)[0]  # Assuming the model output is a single value

        return {"text": text, "prediction": float(prediction)}
    except Exception as e:
        # Handle exceptions gracefully
         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__=="__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8080)

## to run the fastapi without docker :
## python -m uvicorn main:app --reload
