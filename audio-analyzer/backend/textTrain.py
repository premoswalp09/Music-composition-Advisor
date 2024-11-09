import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Dict, Tuple
import os
import pickle
from transformers import BertTokenizer



class TextDataset(Dataset):
    def __init__(self, data_path: str = 'Final_cleaned_genres_output.csv', tokenizer=None, max_length: int = 512):
        self.data = pd.read_csv(data_path)
        self.max_length = max_length

        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

            # Initialize label encoders
        self.genre_encoder = LabelEncoder()
        self.sentiment_encoder = LabelEncoder()

        # Encode labels
        self.genres = self.genre_encoder.fit_transform(self.data['CleanedGenre'].values)
        self.sentiments = self.sentiment_encoder.fit_transform(self.data['Label'].values)

        # Save encoders
        if not os.path.exists('encoders'):
            os.makedirs('encoders')
        with open('encoders/genre_encoder.pkl', 'wb') as f:
            pickle.dump(self.genre_encoder, f)
        with open('encoders/sentiment_encoder.pkl', 'wb') as f:
            pickle.dump(self.sentiment_encoder, f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = str(self.data.iloc[idx]['Lyrics'])  # Assuming 'lyrics' column exists

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'genre_label': torch.tensor(self.genres[idx], dtype=torch.long),
            'sentiment_label': torch.tensor(self.sentiments[idx], dtype=torch.long)
        }


class TextClassifier(nn.Module):
    def __init__(self, num_genres: int, num_sentiments: int):
        super(TextClassifier, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT parameters (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Classification heads
        self.genre_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_genres)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_sentiments)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Get predictions
        genre_output = self.genre_classifier(pooled_output)
        sentiment_output = self.sentiment_classifier(pooled_output)

        return genre_output, sentiment_output


def train_text_model(train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                model: TextClassifier,
                device: torch.device,
                num_epochs: int = 50,
                learning_rate: float = 2e-5) -> None:
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize loss functions
    genre_criterion = nn.CrossEntropyLoss()
    sentiment_criterion = nn.CrossEntropyLoss()

    # Track best validation loss
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            genre_labels = batch['genre_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            genre_output, sentiment_output = model(input_ids, attention_mask)

            # Calculate losses
            genre_loss = genre_criterion(genre_output, genre_labels)
            sentiment_loss = sentiment_criterion(sentiment_output, sentiment_labels)
            total_loss = genre_loss + sentiment_loss

            # Backward pass
            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()

            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                genre_labels = batch['genre_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                genre_output, sentiment_output = model(input_ids, attention_mask)

                genre_loss = genre_criterion(genre_output, genre_labels)
                sentiment_loss = sentiment_criterion(sentiment_output, sentiment_labels)
                total_loss = genre_loss + sentiment_loss

                total_val_loss += total_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_text_model.pth')
            print('Saved best model checkpoint')


# def predict_sentiment_and_genre(text: str,
#                                 genre: str,
#                                 model: TextClassifier,
#                                 tokenizer: BertTokenizer,
#                                 genre_encoder: LabelEncoder,
#                                 sentiment_encoder: LabelEncoder,
#                                 device: torch.device) -> Dict[str, str]:
#     """
#     Predict sentiment and genre for given text
#     """
#     model.eval()
#
#     # Tokenize input text
#     encoding = tokenizer(
#         text,
#         add_special_tokens=True,
#         max_length=512,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
#
#     with torch.no_grad():
#         genre_output, sentiment_output = model(input_ids, attention_mask)
#
#         genre_pred = torch.argmax(genre_output, dim=1)
#         sentiment_pred = torch.argmax(sentiment_output, dim=1)
#
#         predicted_genre = genre_encoder.inverse_transform([genre_pred.item()])[0]
#         predicted_sentiment = sentiment_encoder.inverse_transform([sentiment_pred.item()])[0]
#
#     return {
#         "genre": predicted_genre,
#         "sentiment": predicted_sentiment
#     }
#
#
# def load_model_and_encoders(device: torch.device) -> Tuple[TextClassifier, BertTokenizer, LabelEncoder, LabelEncoder]:
#     """
#     Load the trained model and encoders
#     """
#     # Load encoders
#     with open('encoders/genre_encoder.pkl', 'rb') as f:
#         genre_encoder = pickle.load(f)
#     with open('encoders/sentiment_encoder.pkl', 'rb') as f:
#         sentiment_encoder = pickle.load(f)
#
#         # Initialize tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#     # Initialize and load model
#     model = TextClassifier(
#         num_genres=len(genre_encoder.classes_),
#         num_sentiments=len(sentiment_encoder.classes_)
#     ).to(device)
#
#     model.load_state_dict(torch.load('best_text_model.pth'))
#     model.eval()
#
#     return model, tokenizer, genre_encoder, sentiment_encoder
#


def predict_sentiment_and_genre(text: str, model: TextClassifier, tokenizer: BertTokenizer,
                                genre_encoder: LabelEncoder, sentiment_encoder: LabelEncoder,
                                device: torch.device) -> Dict[str, str]:
    """
    Predict genre and sentiment for given text
    """
    # Ensure model is in eval mode
    model.eval()

    # Tokenize text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get predictions
    with torch.no_grad():
        genre_logits, sentiment_logits = model(input_ids, attention_mask)

        genre_pred = torch.argmax(genre_logits, dim=1)
        sentiment_pred = torch.argmax(sentiment_logits, dim=1)

        predicted_genre = genre_encoder.inverse_transform([genre_pred.item()])[0]
        predicted_sentiment = sentiment_encoder.inverse_transform([sentiment_pred.item()])[0]

    sentiment_encoding = {"Negative": 0, "Neutral": 1, "Positive": 2}

    return {
        "genre": predicted_genre,
        "sentiment": sentiment_encoding[predicted_sentiment]
    }
    # return {
    #     "genre": predicted_genre,
    #     "sentiment": predicted_sentiment
    # }


def load_model_and_encoders(model_path: str = 'best_text_model.pth') -> Tuple[
    TextClassifier, BertTokenizer, LabelEncoder, LabelEncoder]:
    """
    Load the trained model and necessary encoders
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load encoders
    with open('encoders/genre_encoder.pkl', 'rb') as f:
        genre_encoder = pickle.load(f)
    with open('encoders/sentiment_encoder.pkl', 'rb') as f:
        sentiment_encoder = pickle.load(f)

        # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize model
    model = TextClassifier(
        num_genres=len(genre_encoder.classes_),
        num_sentiments=len(sentiment_encoder.classes_)
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer, genre_encoder, sentiment_encoder


# Example usage:

# if __name__ == "__main__":
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load data
#     df = pd.read_csv('your_data.csv')  # Replace with your data path
#
#     # Create dataset and split into train/val
#     dataset = TextDataset(df)
#
#     # Calculate split sizes
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#
#     # Split dataset
#     train_dataset, val_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size]
#     )
#
#     # Create dataloaders
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=8,
#         shuffle=True
#     )
#
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=8
#     )
#
#     # Initialize model
#     model = TextClassifier(
#         num_genres=len(dataset.genre_encoder.classes_),
#         num_sentiments=len(dataset.sentiment_encoder.classes_)
#     ).to(device)
#
#     # Train model
#     train_model(train_dataloader, val_dataloader, model, device)
#
#     # Example prediction
#     sample_text = "This is a sample lyrics text"
#     sample_genre = "Pop"
#
#     result = predict_sentiment_and_genre(
#         sample_text,
#         sample_genre,
#         model,
#         dataset.tokenizer,
#         dataset.genre_encoder,
#         dataset.sentiment_encoder,
#         device
#     )
#
#     print("Prediction:", result)


def make_predict(lyric):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, genre_encoder, sentiment_encoder = load_model_and_encoders()

    # Example text for prediction
    # sample_text = "would without smart mouth Drawing kicking got head spinning no kidding cannot pin going beautiful mind magical mystery ride dizzy not know hit alright head's water breathing fine crazy mind Loves Love curves edges perfect imperfections Give give end beginning Even lose winning give give ooh many times tell Even crying beautiful world beating around every mood downfall muse worst distraction rhythm blues cannot stop singing ringing head head's water breathing fine crazy mind Loves Love curves edges perfect imperfections Give give end beginning Even lose winning give give ooh Give oh Cards table showing hearts Risking though hard Loves Love curves edges perfect imperfections Give give end beginning Even lose winning give give give give ooh"

    # Make prediction
    result = predict_sentiment_and_genre(
        text=lyric,
        model=model,
        tokenizer=tokenizer,
        genre_encoder=genre_encoder,
        sentiment_encoder=sentiment_encoder,
        device=device
    )

    sentiment_encoding = {"Negative": 0, "Neutral": 1, "Positive": 2}
    print("Prediction:", result)
    return result




if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = TextDataset('Final_cleaned_genres_output.csv')

    # Calculate split sizes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8
    )

    # Initialize model
    model = TextClassifier(
        num_genres=len(dataset.genre_encoder.classes_),
        num_sentiments=len(dataset.sentiment_encoder.classes_)
    ).to(device)

    # Train model
    train_text_model(train_dataloader, val_dataloader, model, device)


