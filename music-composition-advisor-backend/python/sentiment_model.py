#This is demo code which needs to be replace by original model inference file.

import sys
import json

def predict_sentiment(lyrics, genre):
    # Placeholder for the model inference
    # Load your trained model here and make predictions based on `lyrics` and `genre`
    # For now, we'll return a placeholder response
    if "love" in lyrics.lower():
        return "Positive"
    else:
        return "Negative"

if __name__ == "__main__":
    lyrics = sys.argv[1]
    genre = sys.argv[2]
    prediction = predict_sentiment(lyrics, genre)
    print(prediction)
