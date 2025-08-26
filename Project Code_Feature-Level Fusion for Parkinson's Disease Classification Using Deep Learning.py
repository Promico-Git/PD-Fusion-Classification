#%%
'''from google.colab import drive
drive.mount('/content/drive')'''
#%%
data = 'Datasets'
#%%
'''data = "/content/drive/MyDrive/Projects/Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning/Datasets"
[print(os.path.join(data, item)) for item in os.listdir(data)]'''
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
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from glob import glob
import random
import torchvision.transforms as T
#%%
# Set random seeds for reproducibility
random.seed(50)
torch.manual_seed(50)
np.random.seed(50)
#%%
"""### Data Loading"""
def load_balanced_data_paths(mri_dir, audio_dir):
    mri_paths, audio_paths, labels = [], [], []
    for label, group in enumerate(['healthy', 'patient']):
        mri_group = sorted(glob(os.path.join(mri_dir, group, '*')))
        audio_group = sorted(glob(os.path.join(audio_dir, group, '*')))

        # Resample the smaller dataset to match the larger one
        if len(mri_group) > len(audio_group):
            audio_group.extend(random.choices(audio_group, k=len(mri_group) - len(audio_group)))
        elif len(audio_group) > len(mri_group):
            mri_group.extend(random.choices(mri_group, k=len(audio_group) - len(mri_group)))

        mri_paths.extend(mri_group)
        audio_paths.extend(audio_group)
        labels.extend([label] * len(mri_group))  # Assign label based on the group (0: healthy, 1: patient)
    return mri_paths, audio_paths, labels # Return the paths and corresponding labels
#%%
"""### Preprocessing"""
def preprocess_mri(image_path, augment=False):
    img = nib.load(image_path).get_fdata()

    # Clip extreme values to normalize contrast
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    img = (img - img.min()) / (img.max() - img.min()) # Normalize

    # Resize depth to 64 slices
    target_depth = 64
    resized_img = np.zeros((64, 64, target_depth), dtype=np.float32)
    depth = min(img.shape[2], target_depth)

    for i in range(depth):
        resized_img[:, :, i] = cv2.resize(img[:, :, i], (64, 64), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(resized_img, (2, 0, 1)) # Shape: (D, H, W)

    if augment:
        # Apply random augmentations
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        img = transform(torch.tensor(img).float())
        # Convert the tensor back to NumPy and then to float32
        img = img.numpy().astype(np.float32)

    return img.astype(np.float32)

def preprocess_audio(audio_path, augment=False):
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

    # Generate mel spectrogram and convert to decibels
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spect_db = librosa.power_to_db(spect, ref=np.max)

    # Resize and normalize
    spect_db = cv2.resize(spect_db, (128, 128))
    spect_db = (spect_db - spect_db.min()) / (spect_db.max() - spect_db.min() + 1e-6)

    return spect_db.astype(np.float32)
#%%
"""### Dataset"""
class ParkinsonDataset(Dataset):
    def __init__(self, mri_paths, audio_paths, labels, augment=False):
        self.mri_paths = mri_paths
        self.audio_paths = audio_paths
        self.labels = labels
        self.augment = augment


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if self.augment == True:
            mri = preprocess_mri(self.mri_paths[idx], augment=False)
            audio = preprocess_audio(self.audio_paths[idx], augment=True)

        else:
            mri = preprocess_mri(self.mri_paths[idx], augment=False)
            audio = preprocess_audio(self.audio_paths[idx], augment=False)


        label = self.labels[idx]
        return torch.tensor(mri).unsqueeze(0), torch.tensor(audio).unsqueeze(0), torch.tensor(label, dtype=torch.long)
#%%
"""### Models (MRI & Audio)"""
# MRI Model
class MRI3DCNN(nn.Module):
    def __init__(self):
        super(MRI3DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.fc = nn.Linear(16 * 16 * 16 * 16, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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
        self.fc = nn.Linear(32 * 32 * 32, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
#%%
"""### Fusion Classifier"""

class FusionClassifier(nn.Module):
    def __init__(self):
        super(FusionClassifier, self).__init__()
        self.mri_branch = MRI3DCNN()
        self.audio_branch = AudioCNN()


        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, mri, audio):
        mri_feat = self.mri_branch(mri)
        audio_feat = self.audio_branch(audio)

        fused = torch.cat([mri_feat, audio_feat], dim=1)
        return self.classifier(fused)
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
        for mri, audio, label in dataloader:
            mri, audio, label = mri.to(device), audio.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(mri, audio) if name == 'fusion' else model(mri if name == 'mri' else audio)
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
            for mri, audio, label in dataloader:
                mri, audio, label = mri.to(device), audio.to(device), label.to(device)
                outputs = model(mri, audio) if name == 'fusion' else model(mri if name == 'mri' else audio)
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

def evaluate_model(model, dataloader, device, model_type):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for mri, audio, label in dataloader:
            mri, audio, label = mri.to(device), audio.to(device), label.to(device)

            if model_type == 'fusion':
                outputs = model(mri, audio)
            elif model_type == 'mri':
                outputs = model(mri) # Pass only MRI data for MRI model
            elif model_type == 'audio':
                outputs = model(audio) # Pass only Audio data for Audio model
            else:
                raise ValueError("Invalid model_type. Must be 'mri', 'audio', or 'fusion'.")

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(label.cpu().numpy())
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

    mri_paths, audio_paths, labels = load_balanced_data_paths(f"{data}/MRI", f"{data}/Audio")
    train_mri, test_mri, train_audio, test_audio, train_labels, test_labels = train_test_split(
        mri_paths, audio_paths, labels, test_size=0.2, stratify=labels)

    # Enable augmentation for training
    train_data = ParkinsonDataset(train_mri, train_audio, train_labels, augment=True)
    test_data = ParkinsonDataset(test_mri, test_audio, test_labels, augment=False)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrain MRI
    print("\n--- Pretraining MRI Model ---")
    mri_model = MRI3DCNN().to(device)
    mri_optimizer = torch.optim.Adam(mri_model.parameters(), lr=1e-3)
    mri_model = train_model(mri_model, train_loader, mri_optimizer, nn.CrossEntropyLoss(), device, name='mri')
    # Call evaluate_model with 'mri' as model_type
    mri_metrics = evaluate_model(mri_model, test_loader, device, model_type='mri')
    print("MRI Model Metrics:", mri_metrics)

    # Pretrain Audio
    print("\n--- Pretraining Audio Model ---")
    audio_model = AudioCNN().to(device)
    audio_optimizer = torch.optim.Adam(audio_model.parameters(), lr=1e-3)
    audio_model = train_model(audio_model, train_loader, audio_optimizer, nn.CrossEntropyLoss(), device, name='audio')
    # Call evaluate_model with 'audio' as model_type
    audio_metrics = evaluate_model(audio_model, test_loader, device, model_type='audio')
    print("Audio Model Metrics:", audio_metrics)

    # Fusion Training
    print("\n--- Training Fusion Model ---")
    model = FusionClassifier().to(device)
    model.mri_branch.load_state_dict(mri_model.state_dict())
    model.audio_branch.load_state_dict(audio_model.state_dict())
    fusion_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = train_model(model, train_loader, fusion_optimizer, nn.CrossEntropyLoss(), device, name='fusion')

    print("\n--- Evaluating Fusion Model ---")
    # Call evaluate_model with 'fusion' as model_type
    fusion_metrics = evaluate_model(model, test_loader, device, model_type='fusion')
    print("Fusion Model Metrics:", fusion_metrics)
    torch.save(model.state_dict(), "fusion_model.pth")