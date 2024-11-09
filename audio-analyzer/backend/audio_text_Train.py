import torch
from torch.utils.data import Dataset, DataLoader
from audioTrain import AudioDataset, AudioClassifier, predict_genre_from_audio, train_audio_model
from textTrain import TextDataset, TextClassifier, predict_sentiment_and_genre, train_text_model, make_predict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from transformers import BertTokenizer


def save_encoders(genre_encoder, sentiment_encoder, save_dir='encoders'):
    """Save the encoders to disk"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/genre_encoder.pkl', 'wb') as f:
        pickle.dump(genre_encoder, f)

    with open(f'{save_dir}/sentiment_encoder.pkl', 'wb') as f:
        pickle.dump(sentiment_encoder, f)


def load_encoders(load_dir='encoders'):
    """Load the encoders from disk"""
    with open(f'{load_dir}/genre_encoder.pkl', 'rb') as f:
        genre_encoder = pickle.load(f)

    with open(f'{load_dir}/sentiment_encoder.pkl', 'rb') as f:
        sentiment_encoder = pickle.load(f)

    return genre_encoder, sentiment_encoder


def train_combined_models(data_path='Final_cleaned_genres_output.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device:", device)
    # Create datasets
    text_dataset = TextDataset(data_path)
    audio_dataset = AudioDataset(data_path, genre_encoder=text_dataset.genre_encoder)

    # Save encoders
    save_encoders(text_dataset.genre_encoder, text_dataset.sentiment_encoder)

    # Split datasets
    train_size_text = int(0.8 * len(text_dataset))
    val_size_text = len(text_dataset) - train_size_text
    train_text, val_text = torch.utils.data.random_split(text_dataset, [train_size_text, val_size_text])

    train_size_audio = int(0.7 * len(audio_dataset))
    val_size_audio = len(audio_dataset) - train_size_audio
    train_audio, val_audio = torch.utils.data.random_split(audio_dataset, [train_size_audio, val_size_audio])

    # Create dataloaders
    train_text_loader = DataLoader(train_text, batch_size=8, shuffle=True)
    val_text_loader = DataLoader(val_text, batch_size=8)

    train_audio_loader = DataLoader(train_audio, batch_size=8, shuffle=True)
    val_audio_loader = DataLoader(val_audio, batch_size=8)

    # Initialize models
    num_genres = len(text_dataset.genre_encoder.classes_)
    num_sentiments = len(text_dataset.sentiment_encoder.classes_)

    text_model = TextClassifier(num_genres, num_sentiments).to(device)
    audio_model = AudioClassifier(num_genres).to(device)

    # Train models
    print("Training Text Model...")
    train_text_model(train_text_loader, val_text_loader, text_model, device)

    print("\nTraining Audio Model...")
    train_audio_model(train_audio_loader, val_audio_loader, audio_model, device)

    return text_model, audio_model, text_dataset.genre_encoder, text_dataset.sentiment_encoder
    # return audio_model, text_dataset.genre_encoder, text_dataset.sentiment_encoder


def train_audio_models(data_path='Final_cleaned_genres_output.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device:", device)
    # Create datasets
    text_dataset = TextDataset(data_path)
    audio_dataset = AudioDataset(data_path, genre_encoder=text_dataset.genre_encoder)

    save_encoders(text_dataset.genre_encoder, text_dataset.sentiment_encoder)

    # Split datasets
    train_size_text = int(0.8 * len(text_dataset))
    val_size_text = len(text_dataset) - train_size_text
    train_text, val_text = torch.utils.data.random_split(text_dataset, [train_size_text, val_size_text])

    train_size_audio = int(0.7 * len(audio_dataset))
    val_size_audio = len(audio_dataset) - train_size_audio
    train_audio, val_audio = torch.utils.data.random_split(audio_dataset, [train_size_audio, val_size_audio])

    # Create dataloaders
    train_text_loader = DataLoader(train_text, batch_size=8, shuffle=True)
    val_text_loader = DataLoader(val_text, batch_size=8)

    train_audio_loader = DataLoader(train_audio, batch_size=16, shuffle=True)
    val_audio_loader = DataLoader(val_audio, batch_size=16)

    # Initialize models
    num_genres = len(text_dataset.genre_encoder.classes_)
    num_sentiments = len(text_dataset.sentiment_encoder.classes_)

    text_model = TextClassifier(num_genres, num_sentiments).to(device)
    audio_model = AudioClassifier(num_genres).to(device)

    # Train models
    print("Training Text Model...")
    # train_text_model(train_text_loader, val_text_loader, text_model, device)

    print("\nTraining Audio Model...")
    train_audio_model(train_audio_loader, val_audio_loader, audio_model, device)

    # return text_model, audio_model, text_dataset.genre_encoder, text_dataset.sentiment_encoder
    return audio_model, text_dataset.genre_encoder, text_dataset.sentiment_encoder


def predict_complete(text, audio_file):
    """
    Make predictions using both text and audio models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load encoders
    genre_encoder, sentiment_encoder = load_encoders()

    # Load models
    num_genres = len(genre_encoder.classes_)
    num_sentiments = len(sentiment_encoder.classes_)

    text_model = TextClassifier(num_genres, num_sentiments).to(device)
    audio_model = AudioClassifier(num_genres).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_model.load_state_dict(torch.load('best_text_model.pth'))
    audio_model.load_state_dict(torch.load('best_audio_model.pth'))

    # Make predictions
    text_results = predict_sentiment_and_genre(
        text,  text_model, tokenizer, genre_encoder, sentiment_encoder, device
    )

    audio_results = predict_genre_from_audio(
        audio_file, audio_model, genre_encoder, device
    )

    return {
        "text_analysis": text_results,
        "audio_analysis": audio_results
    }

def predict_complete_lyrics(text):
    """
    Make predictions using both text and audio models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load encoders
    genre_encoder, sentiment_encoder = load_encoders()

    # Load models
    num_genres = len(genre_encoder.classes_)
    num_sentiments = len(sentiment_encoder.classes_)

    text_model = TextClassifier(num_genres, num_sentiments).to(device)
    # audio_model = AudioClassifier(num_genres).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_model.load_state_dict(torch.load('best_text_model.pth'))
    # audio_model.load_state_dict(torch.load('best_audio_model.pth'))

    # Make predictions
    text_results = predict_sentiment_and_genre(
        text,  text_model, tokenizer, genre_encoder, sentiment_encoder, device
    )

    # audio_results = predict_genre_from_audio(
    #     audio_file, audio_model, genre_encoder, device
    # )

    return {
        "text_analysis": text_results,
        # "audio_analysis": audio_results
    }


def predict_audio(audio_file):
    """
    Make predictions using both text and audio models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load encoders
    genre_encoder, sentiment_encoder = load_encoders()

    # Load models
    num_genres = len(genre_encoder.classes_)
    num_sentiments = len(sentiment_encoder.classes_)

    text_model = TextClassifier(num_genres, num_sentiments).to(device)
    audio_model = AudioClassifier(num_genres).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_model.load_state_dict(torch.load('best_text_model.pth'))
    audio_model.load_state_dict(torch.load('best_audio_model.pth'))

    audio_results = predict_genre_from_audio(
        audio_file, audio_model, genre_encoder, device
    )

    return {
        "audio_analysis": audio_results
    }



if __name__ == "__main__":
    # Train models
    # text_model, audio_model, genre_encoder, sentiment_encoder = train_combined_models()
    # audio_model, genre_encoder, sentiment_encoder = train_combined_models()
    audio_model, genre_encoder, sentiment_encoder = train_audio_models()

    # Example prediction
    # sample_text = "I've been reading books of old	The legends and the myths	Achilles and his gold	Hercules and his gifts	Spiderman's control	And Batman with his fists	And clearly I don't see myself upon that listBut she said, where'd you wanna go?	How much you wanna risk?	I'm not looking for somebody	With some superhuman gifts	Some superhero	Some fairytale bliss	Just something I can turn to	Somebody I can kissI want something just like this	Doo-doo-doo, doo-doo-doo	Doo-doo-doo, doo-doo	Doo-doo-doo, doo-doo-doo	Oh, I want something just like this	Doo-doo-doo, doo-doo-doo	Doo-doo-doo, doo-doo	Doo-doo-doo, doo-doo-dooOh, I want something just like this	I want something just like thisI've been reading books of old	The legends and the myths	The testaments they told	The moon and its eclipse	And Superman unrolls	A suit before he lifts	But I'm not the kind of person that it fitsShe said, where'd you wanna go?	How much you wanna risk?	I'm not looking for somebody	With some superhuman gifts	Some superhero	Some fairytale bliss	Just something I can turn to	Somebody I can missI want something just like this	I want something just like thisOh, I want something just like this	Doo-doo-doo, doo-doo-doo	Doo-doo-doo, doo-doo	Doo-doo-doo, doo-doo-doo	Oh, I want something just like this	Doo-doo-doo, doo-doo-doo	Doo-doo-doo, doo-doo	Doo-doo-doo, doo-doo-dooWhere'd you wanna go?	How much you wanna risk?	I'm not looking for somebody	With some superhuman gifts	Some superhero	Some fairytale bliss	Just something I can turn to	Somebody I can kiss	I want something just like thisOh, I want something just like this	Oh, I want something just like this	Oh, I want something just like this"
    # # sample_text = "Kick dust kick dust Ah-ha come week long farming town making money grow Tractors plows flashing lights Backing two lane road take one last lap around sun high goes song come girl kick back Z71 like Cadillac go way not nobody turn cornfield party Pedal floorboard end Four-door burning back road song Park pile Baby watch step better boots Kick dust Back Fill cup Let us tear kick dust (Kick dust kick dust Bar downtown got line people way door Ten dollar drinks packed inside not know waiting Got jar full clear got music ears like goes diesel really want see beautiful people go way not nobody turn cornfield party Pedal floorboard end Four-door burning back road song Park pile Baby watch step better boots Kick dust Let us back Fill cup Let us tear kick dust Come follow neath 32 bridge glad Kick go way not nobody turn cornfield party Pedal floorboard end Four-door burning back road song Park pile Baby watch step Better boots Kick dust kick dust Back Fill cup Let us kick dust"
    # # sample_audio = '../audio_data/Billboard/John Legend - All of Me.mp3'
    # sample_audio = '../audio_data/Billboard/Sam Hunt - Body Like A Back Road.mp3'
    #
    # results = predict_complete(sample_text, sample_audio)
    # print("Results:", results)
    # make_predict(sample_text)
