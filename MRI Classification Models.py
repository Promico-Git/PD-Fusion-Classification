#%%
data = 'Datasets'
#%%
import sys
print(sys.executable)
#%%
# Imports
import os
import numpy as np
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
### Load MRI Data
def load_mri_data(mri_dir):
    mri_paths, labels = [], []
    for label, group in enumerate(['healthy', 'patient']):
        mri_group = sorted(glob(os.path.join(mri_dir, group, '*')))

        mri_group.extend(random.choices(mri_group, k=50))

        mri_paths.extend(mri_group)
        labels.extend([label] * len(mri_group))

    return mri_paths, labels
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
#%%
"""### Dataset"""
class ParkinsonDataset(Dataset):
    def __init__(self, mri_paths, labels, augment=False):
        self.mri_paths = mri_paths
        self.labels = labels
        self.augment = augment


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if self.augment == True:
            mri = preprocess_mri(self.mri_paths[idx], augment=False)

        else:
            mri = preprocess_mri(self.mri_paths[idx], augment=False)


        label = self.labels[idx]
        return torch.tensor(mri).unsqueeze(0), torch.tensor(label, dtype=torch.long)
#%%
"""### CNN Model"""
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
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 16 * 16, 128), # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # Binary Classification
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#%%
# LSTM Model
# This processes 2D slices within a 3D MRI
class CNN_Feature_Extractor(nn.Module):
    def __init__(self):
        super(CNN_Feature_Extractor, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Linear(32 * 16 * 16, 128)

    def forward(self, x):
        x = self.conv2d(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class MRI_CNN_LSTM(nn.Module):
    def __init__(self, cnn_feature_dim, hidden_dim, num_layers, output_dim):
        super(MRI_CNN_LSTM, self).__init__()
        self.cnn_extractor = CNN_Feature_Extractor()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(cnn_feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)

        cnn_features = self.cnn_extractor(x_reshaped)

        lstm_input = cnn_features.view(batch_size, depth, -1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # Pass through LSTM
        out, (hn, cn) = self.lstm(lstm_input, (h0, c0))

        # Take the output from the last time step for classification
        out = self.fc(out[:, -1, :])
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
        for mri, label in dataloader:
            mri, label = mri.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(mri)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = torch.argmax(output, dim=1)
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
            for mri, label in dataloader:
                mri, label = mri.to(device), label.to(device)
                output = model(mri)
                val_loss += criterion(output, label).item()
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
        for mri, label in dataloader:
            mri = mri.to(device)
            outputs = model(mri)
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

    mri_paths, labels = load_mri_data(f"{data}/MRI")
    train_mri, test_mri, train_labels, test_labels = train_test_split(
        mri_paths, labels, test_size=0.2, stratify=labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CNN Model Training
    print("\n--- Training MRI 3DCNN Model ---")
    train_data_cnn = ParkinsonDataset(train_mri, train_labels, augment=True)
    test_data_cnn = ParkinsonDataset(test_mri, test_labels, augment=False)

    train_loader_cnn = DataLoader(train_data_cnn, batch_size=8, shuffle=True)
    test_loader_cnn = DataLoader(test_data_cnn, batch_size=8, shuffle=False)

    mri_cnn_model = MRI3DCNN().to(device)
    cnn_optimizer = torch.optim.AdamW(mri_cnn_model.parameters(), lr=1e-3)
    mri_cnn_model = train_model(mri_cnn_model, train_loader_cnn, cnn_optimizer, nn.CrossEntropyLoss(), device, name='MRI3DCNN', epochs=10)

    print("\n--- Evaluating MRI 3DCNN Model ---")
    cnn_metrics = evaluate_model(mri_cnn_model, test_loader_cnn, device)
    print("MRI 3DCNN Model Metrics:", cnn_metrics)
    torch.save(mri_cnn_model.state_dict(), "mri_cnn_model.pth")


    # CNN-LSTM Model Training
    print("\n--- Training MRI CNN-LSTM Model ---")
    train_data_cnnlstm = ParkinsonDataset(train_mri, train_labels, augment=True)
    test_data_cnnlstm = ParkinsonDataset(test_mri, test_labels, augment=False)

    train_loader_cnnlstm = DataLoader(train_data_cnnlstm, batch_size=8, shuffle=True)
    test_loader_cnnlstm = DataLoader(test_data_cnnlstm, batch_size=8, shuffle=False)


    temp_mri_slice = torch.zeros(1, 1, 64, 64).to(device)
    temp_cnn_extractor = CNN_Feature_Extractor().to(device)
    cnnlstm_cnn_feature_dim = temp_cnn_extractor(temp_mri_slice).shape[1]


    cnnlstm_hidden_dim = 128
    cnnlstm_num_layers = 2
    cnnlstm_output_dim = 2

    mri_cnnlstm_model = MRI_CNN_LSTM(cnnlstm_cnn_feature_dim, cnnlstm_hidden_dim, cnnlstm_num_layers, cnnlstm_output_dim).to(device)
    cnnlstm_optimizer = torch.optim.AdamW(mri_cnnlstm_model.parameters(), lr=1e-3)
    mri_cnnlstm_model = train_model(mri_cnnlstm_model, train_loader_cnnlstm, cnnlstm_optimizer, nn.CrossEntropyLoss(), device, name='MRICNN-LSTM', epochs=10)

    print("\n--- Evaluating MRI CNN-LSTM Model ---")
    cnnlstm_metrics = evaluate_model(mri_cnnlstm_model, test_loader_cnnlstm, device)
    print("MRI CNN-LSTM Model Metrics:", cnnlstm_metrics)
    torch.save(mri_cnnlstm_model.state_dict(), "mri_cnnlstm_model.pth")