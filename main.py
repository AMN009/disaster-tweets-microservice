from fastapi import FastAPI
from pydantic import BaseModel
import model

mdl = model.clf

app = FastAPI()

class Tweet(BaseModel):
    content: str

@app.post("/check-tweet/")
async def check(tweet: Tweet):
    preprocessed_text = [model.text_preprocessing(tweet.content)]
    rep_text = model.text_representation(preprocessed_text)
    result = mdl.predict(rep_text).item()
    return {"disaster": bool(result)}