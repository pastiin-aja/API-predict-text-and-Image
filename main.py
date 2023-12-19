import os
import tensorflow as tf
import pickle
from pydantic import BaseModel
from fastapi import FastAPI, Response, UploadFile,  HTTPException
from easyocr import Reader

# Load the CountVectorizer from the joblib file
loaded_vectorizer = pickle.load(open('./vectorizer.pickle', 'rb'))

# Initialize Model
# If you already put your model in the same folder as this main.py
# You can load .h5 model or any model below this line
model = tf.keras.models.load_model('./model_fraud_v1.h5')

class InputData(BaseModel):
    st: str

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

@app.post("/predict_text")
def predictor(data: InputData):
    # Vectorize the input text
    text_vectorized = loaded_vectorizer.transform([data.st])

    result = model_predict(text_vectorized)
    return {"msg": result.tolist()}  # Convert predictions to a list

@app.post("/extract_and_predict", response_model=PredictionResult)
async def extract_and_predict(data: UploadFile):
    try:
        # Checking if it's an image
        if data.content_type not in ["image/jpeg", "image/png"]:
            return Response(content="File is Not an Image", status_code=400)

        # Save the uploaded file
        with open(data.filename, "wb") as image_file:
            image_file.write(data.file.read())

        # Use EasyOCR to extract text from the image
        reader = Reader(['en', 'id'])  # You can add more languages if needed
        result = reader.readtext(data.filename)

        # Extract text from the result
        text = " ".join([item[1] for item in result])

        # Vectorize the extracted text
        text_vectorized = loaded_vectorizer.transform([text])

        # Make predictions using the loaded model
        prediction = model_predict(text_vectorized)[0]  # Assuming the model output is a single value

        return {"text": text, "prediction": float(prediction)}
    except Exception as e:
        # Handle exceptions gracefully
        return Response(content=f"An error occurred: {str(e)}", status_code=500)


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"),port=int(os.getenv("PORT",8080)),
    log_level="debug")

## to run the fastapi without docker :
## python -m uvicorn main:app --reload