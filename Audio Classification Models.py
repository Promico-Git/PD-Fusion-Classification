#%%
data = 'Datasets'
#%%
import sys
print(sys.executable)
#%%
# Imports
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from glob import glob
import random
#%%
# Set random seeds for reproducibility
random.seed(40)
torch.manual_seed(40)
np.random.seed(40)
#%%
"""### Data Loading"""
def load_audio_data(audio_dir):
    audio_paths, labels = [], []
    for label, group in enumerate(['healthy', 'patient']):
        audio_group = sorted(glob(os.path.join(audio_dir, group, '*')))

        audio_paths.extend(audio_group)
        labels.extend([label] * len(audio_group))  # Assign label based on the group (0: healthy, 1: patient)
    return audio_paths, labels # Return the paths and corresponding labels
#%%
def preprocess_audio(audio_path, augment=False, for_rnn=False):
    y, sr = librosa.load(audio_path, sr=22050) # 22.05 kHz

    if augment:
        # Time stretch
        if random.random() < 0.5:
            rate = random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)

        # Pitch shift
        if random.random() < 0.5:
            n_steps = random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

        # Add noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.005, size=y.shape)
            y = y + noise

    if for_rnn:
        # Feature Extraction for RNN: MFCCs + Delta + Delta-Delta
        n_mfcc = 40 # Number of MFCC coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        features = features.T


        max_seq_len = 200
        if features.shape[0] > max_seq_len:
            features = features[:max_seq_len, :]
        else:
            padding = np.zeros((max_seq_len - features.shape[0], features.shape[1]))
            features = np.vstack((features, padding))

        # Normalize features
        min_val = features.min(axis=0)
        max_val = features.max(axis=0)
        features = (features - min_val) / (max_val - min_val + 1e-6)

        return features.astype(np.float32)

    else:
        # Feature Extraction for CNN: Mel Spectrogram
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spect_db = librosa.power_to_db(spect, ref=np.max)

        # Resize and normalize for CNN
        spect_db = cv2.resize(spect_db, (128, 128))
        spect_db = (spect_db - spect_db.min()) / (spect_db.max() - spect_db.min() + 1e-6)

        return spect_db.astype(np.float32)
#%%
class ParkinsonDataset(Dataset):
    def __init__(self, audio_paths, labels, augment=False, for_rnn=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.augment = augment
        self.for_rnn = for_rnn # New flag

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.augment:
            audio_features = preprocess_audio(self.audio_paths[idx], augment=True, for_rnn=self.for_rnn)
        else:
            audio_features = preprocess_audio(self.audio_paths[idx], augment=False, for_rnn=self.for_rnn)

        label = self.labels[idx]

        # For CNN, it's (1, H, W) -> (channels, height, width)
        # For RNN, it's (seq_len, input_size)
        if not self.for_rnn:
            audio_features = torch.tensor(audio_features).unsqueeze(0)

        return torch.tensor(audio_features), torch.tensor(label, dtype=torch.long)
#%%
"""CNN Model"""
# Audio Model
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
#%%
### RNN Model
class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=True, dropout_rate=0.3):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout_rate if num_layers > 1 else 0) # Dropout applied between layers if num_layers > 1

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Initialize hidden state and cell state for the first time step
        # h_0 and c_0 shape: (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: (batch_size, seq_len, num_directions * hidden_size)
        # (h_n, c_n): (num_layers * num_directions, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Take the output from the last layer (h_n) for classification.
        # h_n contains the hidden state for the last time step of each layer.
        # For bidirectional, the last layer's forward and backward hidden states are concatenated.
        if self.bidirectional:
            # Concatenate the hidden states from the last layer (forward and backward)
            # h_n[-2, :, :] is the last forward hidden state
            # h_n[-1, :, :] is the last backward hidden state
            final_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            final_hidden_state = h_n[-1, :, :] # Last hidden state of the last layer

        final_hidden_state = self.dropout(final_hidden_state)
        out = self.fc(final_hidden_state)
        return out
#%%
"""### Training Function"""
def train_model(model, dataloader, optimizer, criterion, device, epochs=10, name='model', patience=3):
    model.train()
    losses, accuracies, val_losses = [], [], []
    best_loss = float('inf')
    epochs_no_improve = 0

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        model.train()
        for audio, label in dataloader:
            audio, label = audio.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        losses.append(avg_loss)
        accuracies.append(acc)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audio, label in dataloader:
                audio, label = audio.to(device), label.to(device)
                outputs = model(audio)
                val_loss += criterion(outputs, label).item()
        val_loss /= len(dataloader)
        val_losses.append(val_loss)

        # Update LR based on val_loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - {name} Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {acc:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered.")
                break

    # Loss and Accuracy Plotting
    plt.figure()
    plt.plot(losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{name.upper()} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title(f'{name.upper()} Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

    return model
#%%
"""### Evaluation"""

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for audio, label in dataloader:
            audio = audio.to(device)
            outputs = model(audio)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(label.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score)
    }

    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.grid(True)
    plt.show()
    return metrics
#%%
# Main Script
if __name__ == "__main__":

    audio_paths, labels = load_audio_data(f"{data}/Audio")
    train_audio, test_audio, train_labels, test_labels = train_test_split(
        audio_paths, labels, test_size=0.2, stratify=labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CNN Model Training ---
    print("\n--- Training Audio CNN Model ---")
    # For CNN, augment=True, for_rnn=False
    train_data_cnn = ParkinsonDataset(train_audio, train_labels, augment=True, for_rnn=False)
    test_data_cnn = ParkinsonDataset(test_audio, test_labels, augment=False, for_rnn=False)

    train_loader_cnn = DataLoader(train_data_cnn, batch_size=8, shuffle=True)
    test_loader_cnn = DataLoader(test_data_cnn, batch_size=8, shuffle=False)

    audio_cnn_model = AudioCNN().to(device)
    cnn_optimizer = torch.optim.AdamW(audio_cnn_model.parameters(), lr=1e-3)
    audio_cnn_model = train_model(audio_cnn_model, train_loader_cnn, cnn_optimizer, nn.CrossEntropyLoss(), device, name='AudioCNN', epochs=10)

    print("\n--- Evaluating Audio CNN Model ---")
    cnn_metrics = evaluate_model(audio_cnn_model, test_loader_cnn, device)
    print("Audio CNN Model Metrics:", cnn_metrics)
    torch.save(audio_cnn_model.state_dict(), "audio_cnn_model.pth")


    # --- RNN Model Training ---
    print("\n--- Training Audio RNN Model ---")
    # For RNN, augment=True, for_rnn=True
    train_data_rnn = ParkinsonDataset(train_audio, train_labels, augment=True, for_rnn=True)
    test_data_rnn = ParkinsonDataset(test_audio, test_labels, augment=False, for_rnn=True)

    train_loader_rnn = DataLoader(train_data_rnn, batch_size=8, shuffle=True)
    test_loader_rnn = DataLoader(test_data_rnn, batch_size=8, shuffle=False)

    temp_features = preprocess_audio(train_audio[0], for_rnn=True)
    rnn_input_size = temp_features.shape[1]
    rnn_max_seq_len = temp_features.shape[0]

    # RNN Model Parameters
    rnn_hidden_size = 128
    rnn_num_layers = 2
    rnn_num_classes = 2 # Binary classification for (healthy/patient)
    rnn_dropout_rate = 0.3

    audio_rnn_model = AudioRNN(rnn_input_size, rnn_hidden_size, rnn_num_layers, rnn_num_classes, dropout_rate=rnn_dropout_rate).to(device)
    rnn_optimizer = torch.optim.AdamW(audio_rnn_model.parameters(), lr=1e-3)
    audio_rnn_model = train_model(audio_rnn_model, train_loader_rnn, rnn_optimizer, nn.CrossEntropyLoss(), device, name='AudioRNN', epochs=10)

    print("\n--- Evaluating Audio RNN Model ---")
    rnn_metrics = evaluate_model(audio_rnn_model, test_loader_rnn, device)
    print("Audio RNN Model Metrics:", rnn_metrics)
    torch.save(audio_rnn_model.state_dict(), "audio_rnn_model.pth")

