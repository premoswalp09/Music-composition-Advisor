import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# Load your model and tokenizer (replace 'path/to/save/model' with the correct path)
model_path = os.path.join(os.path.dirname(__file__), 'model')
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Function to make predictions
# def predict_sentiment(lyrics, genre):
#     # Combine genre and lyrics as model input
#     text_input = f"{genre} {lyrics}"
    
#     # Tokenize the input
#     inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

#     # Perform inference
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predicted_label = torch.argmax(outputs.logits, dim=1).item()
    
#     # Map prediction to labels
#     labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
#     label = labels.get(predicted_label, "Unknown")

#     return label

def predict_sentiment(lyrics, genre):
    inputs = tokenizer(
        genre,
        lyrics,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True  # Important for distinguishing genre and lyrics
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    label_mapping = {0: "Positive", 1: "Negative", 2: "Neutral"}

    predicted_label = label_mapping[predicted_class]
    return predicted_label

# def predict_sentiment(genre, lyrics, model, tokenizer):  
#     # Tokenize genre and lyrics as separate inputs  
#     inputs = tokenizer(  
#         genre,  
#         lyrics,  
#         return_tensors="pt",  
#         padding=True,  
#         truncation=True,  
#         max_length=512,  
#         return_token_type_ids=True  # Important for distinguishing genre and lyrics  
#     )  
#     with torch.no_grad():  
#         outputs = model(**inputs)  

#     predicted_class = torch.argmax(outputs.logits, dim=1).item()  

#     label_mapping = {0: "Positive", 1: "Negative", 2: "Neutral"}  

#     predicted_label = label_mapping[predicted_class]  
#     return predicted_label  


# lyrics =  "Uh-huh! Iggy Iggs! got one problem girl lem"
# genre = "Contemporary R&B, R&B/Soul"

if __name__ == "__main__":
    lyrics = sys.argv[1]
    genre = sys.argv[2]
    prediction = predict_sentiment(lyrics, genre)
    
    # Print the result as JSON to send it back to Node.js
    print(json.dumps({"label": prediction}))
