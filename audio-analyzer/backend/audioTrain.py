import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES = SAMPLE_RATE * DURATION
MEL_BANDS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

writer = SummaryWriter('runs/experiment_1')

class AudioDataset(Dataset):
    def __init__(self, data_path='Final_cleaned_genres_output.csv', genre_encoder=None):
        self.data = pd.read_csv(data_path)

        # Initialize or use provided genre encoder
        if genre_encoder is None:
            self.genre_encoder = LabelEncoder()
            self.genres = self.genre_encoder.fit_transform(self.data['CleanedGenre'].values)
        else:
            self.genre_encoder = genre_encoder
            self.genres = self.genre_encoder.transform(self.data['CleanedGenre'].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['file_path']
        genre_label = self.genres[idx]

        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, duration=30)  # Load 30 seconds

            target_length = 30 * sr  # 30 seconds * sample rate
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]


                # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=128,
                fmax=8000
            )

            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

            # Convert to tensor
            mel_spec_tensor = torch.FloatTensor(mel_spec_db)

            mel_spec_tensor = mel_spec_tensor.unsqueeze(0)  # Shape: (1, n_mels, time)

            # Debug prints (only for first few items)
            if idx < 3:  # Print for first 3 items
                print(f"Audio file: {audio_path}")
                print(f"Audio length: {len(audio)}")
                print(f"Mel spectrogram shape: {mel_spec_tensor.shape}")
                print(f"Mel spectrogram range: [{mel_spec_tensor.min():.2f}, {mel_spec_tensor.max():.2f}]")
                print(f"Genre label: {genre_label}")

            return {
                'audio': mel_spec_tensor,
                'genre_label': torch.tensor(genre_label, dtype=torch.long)
            }

        except Exception as e:
            print(f"Error loading file {audio_path}: {str(e)}")
            # Return a zero tensor of appropriate size in case of error
            return {
                'audio': torch.zeros((128, 1292)),  # Mel spectrogram size
                'genre_label': torch.tensor(genre_label, dtype=torch.long)
            }

class AudioClassifier(nn.Module):
    def __init__(self, num_genres):
        super(AudioClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # # Pooling and activation
        # self.pool = nn.MaxPool2d(2, 2)
        # self.relu = nn.ReLU()

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Activation
        self.activation = nn.LeakyReLU(0.01)

        # Dropout
        # self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)



        # Calculate the size of flattened features
        # self._to_linear = None
        # self._calculate_to_linear(torch.zeros((1, 1, 128, 1292)))  # Example input size

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_genres)

        # self.fc1 = nn.Linear(self._to_linear, 512)
        # self.fc2 = nn.Linear(512, num_genres)

    def _calculate_to_linear(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

            # First block
        identity1 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Second block with residual
        identity2 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        if identity2.size(1) != x.size(1):
            identity2 = F.conv2d(identity2, torch.ones(x.size(1), identity2.size(1), 1, 1).to(x.device))
        x = x + identity2

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout3(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x

    def _initialize_size(self):
            # Forward pass with dummy data to calculate size
            x = torch.randn(1, 1, N_MELS, SAMPLES // HOP_LENGTH + 1)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self._to_linear = x.numel() // x.size(0)


class AudioTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter('runs/audio_classifier')
        self.scaler = torch.cuda.amp.GradScaler()

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

    @staticmethod
    def augment_audio(spectrogram):
        """Apply data augmentation to the spectrogram"""
        # Time masking
        time_mask_param = int(spectrogram.size(-1) * 0.05)
        if time_mask_param > 0:
            mask_start = torch.randint(0, spectrogram.size(-1) - time_mask_param, (1,))
            spectrogram[..., mask_start:mask_start + time_mask_param] = 0

            # Frequency masking
        freq_mask_param = int(spectrogram.size(-2) * 0.05)
        if freq_mask_param > 0:
            mask_start = torch.randint(0, spectrogram.size(-2) - freq_mask_param, (1,))
            spectrogram[..., mask_start:mask_start + freq_mask_param, :] = 0

        return spectrogram

    def train_step(self, data, targets):
        self.model.train()
        data, targets = data.to(self.device), targets.to(self.device)

        # Apply augmentation
        data = self.augment_audio(data)

        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backpropagation with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), outputs

    def validate_step(self, data, targets):
        self.model.eval()
        with torch.no_grad():
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
        return loss.item(), outputs

    def train_epoch(self, train_loader, epoch):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            loss, outputs = self.train_step(data, targets)
            total_loss += loss

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.to(self.device)).sum().item()

            # Log batch statistics
            self.writer.add_scalar('Loss/train_batch', loss, epoch * len(train_loader) + batch_idx)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        # Log epoch statistics
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train_epoch', accuracy, epoch)

        return avg_loss, accuracy

def validate(self, val_loader, epoch):
    """
    Validate the model on the validation set
    """
    self.model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            loss, outputs = self.validate_step(data, targets)
            total_loss += loss

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.to(self.device)).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    # Log validation statistics
    self.writer.add_scalar('Loss/validation', avg_loss, epoch)
    self.writer.add_scalar('Accuracy/validation', accuracy, epoch)

    return avg_loss, accuracy



def save_genre_encoder(genre_encoder, path='genre_encoder.pkl'):
    """Save the genre encoder to disk"""
    with open(path, 'wb') as f:
        pickle.dump(genre_encoder, f)


def load_genre_encoder(path='genre_encoder.pkl'):
    """Load the genre encoder from disk"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def train_audio_model(train_loader, val_loader, model, device, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Progress tracking
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        batch_idx = 0
        # Training phase
        for batch in train_loader:

            # Correctly unpack the dictionary
            features = batch['audio'].to(device)  # Get audio features
            labels = batch['genre_label'].to(device)  # Get genre labels


            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()


        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['audio'].to(device)
                labels = batch['genre_label'].to(device)
                print('freatures:',features.shape)
                print('labels:',labels.shape)
                print(f"Memory used: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f}MB")

                outputs = model(features)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total


        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_audio_model.pth')
            print('Saved best model checkpoint')

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_validation_loss.png')
        plt.close()

        print("\nLoss plot has been saved as 'training_validation_loss.png'")

    return model

def predict_genre_from_audio(audio_path, model, genre_encoder, device):
    """Predict genre for a given audio file"""
    model.eval()

    # Load and process audio
    try:
        audio, sr = librosa.load(audio_path, duration=DURATION, sr=SAMPLE_RATE)
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        else:
            audio = audio[:SAMPLES]

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None

        # Convert to tensor
    mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
    mel_spec_tensor = mel_spec_tensor.to(device)

    with torch.no_grad():
        output = model(mel_spec_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_genre = genre_encoder.inverse_transform([predicted.item()])[0]

    return predicted_genre


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # Example usage
    audio_paths = []  # List of audio file paths
    genres = []  # List of corresponding genres
    #
    # # Create dataset
    dataset = AudioDataset(audio_paths, genres)
    #
    # # Save genre encoder
    save_genre_encoder(dataset.genre_encoder)
    #
    # # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    #
    # # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    #
    # # Initialize model
    # num_genres = len(dataset.genre_encoder.classes_)
    # model = AudioClassifier(num_genres).to(device)
    #
    # # Train model
    # train_audio_model(train_loader, val_loader, model, device)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier(num_genres=7).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = AudioTrainer(model, criterion, optimizer, device)

    # Training loop
    num_epochs = 100
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)

        val_loss, val_acc = validate(val_loader, epoch)

        # Learning rate scheduling
        trainer.scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
