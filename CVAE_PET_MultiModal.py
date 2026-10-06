"""
Created on Wednesday Feb 26 2025

@author: Fatemeh Esmaeili
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, demo_dim, latent_dim):
        super(CVAE, self).__init__()
        self.demo_dim = demo_dim

        # Encoder for modality 1
        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim_1 + demo_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu_1 = nn.Linear(32, latent_dim)
        self.fc_logvar_1 = nn.Linear(32, latent_dim)

        # Encoder for modality 2
        self.encoder_2 = nn.Sequential(
            nn.Linear(input_dim_2 + demo_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu_2 = nn.Linear(32, latent_dim)
        self.fc_logvar_2 = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder_1 = nn.Sequential(
            nn.Linear(latent_dim + demo_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim_1)
        )

        self.decoder_2 = nn.Sequential(
            nn.Linear(latent_dim + demo_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim_2)
        )

    def encode(self, x1, x2, demo):
        # Encode each modality separately
        x1 = torch.cat((x1, demo), dim=1)
        x2 = torch.cat((x2, demo), dim=1)
        hidden_1 = self.encoder_1(x1)
        hidden_2 = self.encoder_2(x2)
        mu_1, logvar_1 = self.fc_mu_1(hidden_1), self.fc_logvar_1(hidden_1)
        mu_2, logvar_2 = self.fc_mu_2(hidden_2), self.fc_logvar_2(hidden_2)
        return mu_1, logvar_1, mu_2, logvar_2

    def product_of_experts(self, mu_list, logvar_list):
        """
        Compute the Product of Experts (PoE) mean and log-variance.

        Args:
            mu_list (list of tensors): Means from individual modalities.
            logvar_list (list of tensors): Log-variances from individual modalities.

        Returns:
            (mu_poe, logvar_poe): PoE mean and log-variance.
        """
        precisions = [torch.exp(-logvar) for logvar in logvar_list] # T_i = exp(-logvar)
        precision_sum = sum(precisions) # Sum of precisions
        variance_poe = 1.0 / precision_sum # Variance
        weighted_mu_sum = sum(mu * precision for mu, precision in zip(mu_list, precisions))
        mu_poe = weighted_mu_sum * variance_poe  # Weighted mean
        logvar_poe = torch.log(variance_poe) # Log-variance
        return mu_poe, logvar_poe

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, demo):
        # Decode into both modalities
        z = torch.cat((z, demo), dim=1)
        recon_1 = self.decoder_1(z)
        recon_2 = self.decoder_2(z)
        return recon_1, recon_2

    def forward(self, x1, x2, demo):
        mu_1, logvar_1, mu_2, logvar_2 = self.encode(x1, x2, demo)
        # Use Product of Experts (PoE) for the shared representation
        mu_poe, logvar_poe = self.product_of_experts([mu_1, mu_2], [logvar_1, logvar_2])
        z = self.reparameterize(mu_poe, logvar_poe)
        recon_1, recon_2 = self.decode(z, demo)
        return recon_1, recon_2, mu_poe, logvar_poe

# Define the MVAE loss function
def cvae_loss(recon_1, recon_2, x1, x2, mu, logvar, beta):
    reconstruction_loss_1 = nn.MSELoss()(recon_1, x1)
    reconstruction_loss_2 = nn.MSELoss()(recon_2, x2)
    #kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss_1 + reconstruction_loss_2 + beta * kld_loss
    # return reconstruction_loss_1 + reconstruction_loss_2 + kld_loss

def loss_recon1_recon2(recon_1, recon_2, x1, x2, mu, logvar):
    reconstruction_loss_1 = nn.MSELoss()(recon_1, x1)
    reconstruction_loss_2 = nn.MSELoss()(recon_2, x2)
    return reconstruction_loss_1, reconstruction_loss_2

# Load datasets
data_hc_modality1 = pd.read_csv('./Data/roi_demo_both_ADNI.csv') # HC data for modality 1
data_hc_modality2 = pd.read_csv('./Data/ADNI_HC_Schaefer100_TianS1_MRI.csv') # HC data for modality 2
data_ad_modality1 = pd.read_csv('./Data/roi_demo_AD_PET_ADNI.csv') # AD data for modality 1
data_ad_modality2 = pd.read_csv('./Data/roi_demo_AD_PETMRI_ADNI.csv') # AD data for modality 2

# Prepare data for both modalities
X1_hc = data_hc_modality1.iloc[:, 5:].values
X2_hc = data_hc_modality2.iloc[:, 1:].values
X1_ad = data_ad_modality1.iloc[:, 6:].values
X2_ad = data_ad_modality2.iloc[:, 122:].values

# Check for NaN or infinite values and clean
for X in [X1_hc, X2_hc, X1_ad, X2_ad]:
    print("Cleaning data: replacing NaN and Inf values.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = np.nan_to_num(X, nan=np.nanmean(X), posinf=np.nanmax(X), neginf=np.nanmin(X))

# We want to plot features 0 through 7
selected_features = [0, 1, 2, 3, 4, 5, 6, 7]  # Indices of features to plot
# selected_features = np.arange (115)

def plot_compare_histograms(data1, data2, title1, title2, feature_indices):
    """
    Plots individual histograms for each dataset and an overlayed histogram for each selected feature.

    Parameters:
    - data1, data2: Numpy arrays of shape (226, 116).
    - feature_indices: List of feature indices to plot (max 8 features).
    """
    num_features = len(feature_indices)

    titles = [title1, title2]
    data_sets = [data1, data2]
    colors = ["blue", "cyan"]

    fig, axes = plt.subplots(num_features, 3, figsize=(25, num_features * 3))  # 3 columns per feature
    fig.suptitle("Histogram Comparisons with Overlay for Each Feature", fontsize=16)

    for row_idx, feature in enumerate(feature_indices):
        # Plot individual histograms
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            # Check if data is 1D or 2D
            if data_sets[col_idx].ndim == 1:
                data_to_plot = data_sets[col_idx]
            else:
                data_to_plot = data_sets[col_idx][:, feature]

            ax.hist(data_to_plot, bins=20, alpha=0.7, color=colors[col_idx])
            ax.set_title(f"Feature {feature+1} - {titles[col_idx]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        # Plot overlayed histogram in the last column
        ax = axes[row_idx, 2]
        for data, color, label in zip(data_sets, colors, titles):
            if data.ndim == 1:
                data_to_plot = data
            else:
                data_to_plot = data[:, feature]
            ax.hist(data_to_plot, bins=20, alpha=0.5, color=color, label=label)
        ax.set_title(f"Feature {feature+1} - Overlay")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

title1="PET data (Healthy Control)"
title2= "MRI data (Healthy Control)"
plot_compare_histograms(X1_hc, X2_hc, title1, title2, selected_features)

title1="PET data (Alzheimer Disease)"
title2= "MRI data (Alzheimer Disease)"
plot_compare_histograms(X1_ad, X2_ad, title1, title2, selected_features)

hc_age = data_hc_modality1['Age'].values
ad_age = data_ad_modality1['Age'].values
title1= "Age (Healthy Control)"
title2= "Age (Alzheimer Disease)"
plot_compare_histograms(hc_age, ad_age, title1, title2, selected_features)

def age_one_hot_encode(ages, min_age, max_age, step=5):
    bins = list(range(min_age, max_age + step, step))
    age_indices = np.digitize(ages, bins) - 1
    one_hot_encoded = np.eye(len(bins))[age_indices]
    return one_hot_encoded

# Define min and max age
hc_age = data_hc_modality1['Age'].values
min_hc_age = int(np.min(hc_age))
max_hc_age = int(np.max(hc_age))
print(min_hc_age, max_hc_age)

# Define min and max age
ad_age = data_ad_modality1['Age'].values
min_ad_age = int(np.min(ad_age))
max_ad_age = int(np.max(ad_age))
print(min_ad_age, max_ad_age)

# Calculate Overal HC and AD min and max age
min_age = min(min_hc_age, min_ad_age)
max_age = max(max_hc_age, max_ad_age)
print(min_age, max_age)

def sex_one_hot_encode(sex_array):
    return np.array([[1, 0] if sex == 'F' else [0, 1] for sex in sex_array])

# Apply one-hot encoding
hc_age_one_hot = age_one_hot_encode(hc_age, min_age, max_age, step=5)
ad_age_one_hot = age_one_hot_encode(ad_age, min_age, max_age, step=5)

# Extract the 'Age' column
hc_sex = data_hc_modality1['Sex'].values
ad_sex = data_ad_modality1['Sex'].values
hc_sex_one_hot = sex_one_hot_encode(hc_sex)
ad_sex_one_hot = sex_one_hot_encode(ad_sex)

# Combine into final one-hot vector
demographic_features_hc = np.concatenate((hc_age_one_hot, hc_sex_one_hot), axis=1)
demographic_features_ad = np.concatenate((ad_age_one_hot, ad_sex_one_hot), axis=1)

# Convert to a PyTorch tensor
demo_hc_tensor = torch.tensor(demographic_features_hc, dtype=torch.float32)
demo_ad_tensor = torch.tensor(demographic_features_ad, dtype=torch.float32)

# Print shape to verify
print("One-hot encoded HC demo tensor shape:", demo_hc_tensor.shape)
print("One-hot encoded AD demo tensor shape:", demo_ad_tensor.shape)

# Print results
print(demo_hc_tensor)
print(demo_ad_tensor)

# Combine features with demographics
X1_hc_tensor = torch.tensor(X1_hc, dtype=torch.float32)
X2_hc_tensor = torch.tensor(X2_hc, dtype=torch.float32)
print(X1_hc_tensor.shape)
print(X2_hc_tensor.shape)

print(X1_ad.shape)
print(X2_ad.shape)

# MVAE model parameters
input_dim_1 = X1_hc_tensor.shape[1]
input_dim_2 = X2_hc_tensor.shape[1]
roi_dim = input_dim_1
demo_dim = demo_hc_tensor.shape[1]
print(demo_dim)
print(input_dim_1, input_dim_2)
latent_dim = 32
# batch_size = 32
batch_size = 1
epochs = 200
learning_rate = 1e-3
beta = 0.001

# Define 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=False)

# Training loop with 5-fold cross-validation
fold_train_losses = []
fold_test_losses = []
fold_train_losses_recon1 = []
fold_test_losses_recon1 = []
fold_train_losses_recon2 = []
fold_test_losses_recon2 = []

# Train and calculate dij for each fold
for fold, (train_idx, test_idx) in enumerate(kfold.split(X1_hc_tensor)):
    print(f"Fold {fold + 1}")

    # Split Healthy Control data into train and validation sets
    train_X1 = X1_hc_tensor[train_idx]
    test_X1 = X1_hc_tensor[test_idx]
    train_X2 = X2_hc_tensor[train_idx]
    test_X2 = X2_hc_tensor[test_idx]
    train_demo = demo_hc_tensor[train_idx]
    test_demo = demo_hc_tensor[test_idx]

    train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_demo), batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_demo), batch_size = batch_size, shuffle = False)

    # Initialize the model for each fold
    model = CVAE(input_dim_1, input_dim_2, demo_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train_loss_history = []
    train_loss_recon1_history = []
    train_loss_recon2_history = []

    test_loss_history = []
    test_loss_recon1_history = []
    test_loss_recon2_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_recon_1 = 0.0
        train_loss_recon_2 = 0.0
        for batch1, batch2, batch_demo in train_loader:
            optimizer.zero_grad()
            recon_1, recon_2, mu, logvar = model(batch1, batch2, batch_demo)
            loss = cvae_loss(recon_1, recon_2, batch1, batch2, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loss_recon_1, loss_recon_2 = loss_recon1_recon2(recon_1, recon_2, batch1, batch2, mu, logvar)
            train_loss_recon_1 += loss_recon_1.item()
            train_loss_recon_2 += loss_recon_2.item()

        train_loss /= len(train_loader.dataset)
        train_loss_recon_1 /= len(train_loader.dataset)
        train_loss_recon_2 /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        train_loss_recon1_history.append(train_loss_recon_1)
        train_loss_recon2_history.append(train_loss_recon_2)

        model.eval()
        test_loss = 0.0
        test_loss_recon_1 = 0.0
        test_loss_recon_2 = 0.0
        with torch.no_grad():
            for batch1, batch2, batch_demo in test_loader:
                recon_1, recon_2, mu, logvar = model(batch1, batch2, batch_demo)
                loss = cvae_loss(recon_1, recon_2, batch1, batch2, mu, logvar, beta)
                test_loss += loss.item()
                loss_recon_1, loss_recon_2 = loss_recon1_recon2(recon_1, recon_2, batch1, batch2, mu, logvar)
                test_loss_recon_1 += loss_recon_1.item()
                test_loss_recon_2 += loss_recon_2.item()

            test_loss /= len(test_loader.dataset)
            test_loss_recon_1 /= len(test_loader.dataset)
            test_loss_recon_2 /= len(test_loader.dataset)
            test_loss_history.append(test_loss)
            test_loss_recon1_history.append(test_loss_recon_1)
            test_loss_recon2_history.append(test_loss_recon_2)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss Recon1: {train_loss_recon_1:.4f}, Test Loss Recon1: {test_loss_recon_1:.4f}")
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss Recon2: {train_loss_recon_2:.4f}, Test Loss Recon2: {test_loss_recon_2:.4f}")

    fold_train_losses.append(train_loss_history)
    fold_train_losses_recon1.append(train_loss_recon1_history)
    fold_train_losses_recon2.append(train_loss_recon2_history)
    fold_test_losses.append(test_loss_history)
    fold_test_losses_recon1.append(test_loss_recon1_history)
    fold_test_losses_recon2.append(test_loss_recon2_history)

# Plot training and test losses for each fold
plt.figure(figsize=(12, 8))
for fold in range(5):
    plt.plot(fold_train_losses[fold], label=f'Fold {fold + 1} Train Loss')
    plt.plot(fold_test_losses[fold], label=f'Fold {fold + 1} Test Loss', linestyle='--')

plt.title("Training and Test Losses for Each Fold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and test losses for each fold
plt.figure(figsize=(12, 8))
for fold in range(5):
    plt.plot(fold_train_losses_recon1[fold], label=f'Fold {fold + 1} Train Reconstruction1 Loss')
    plt.plot(fold_test_losses_recon1[fold], label=f'Fold {fold + 1} Test Reconstruction1 Loss', linestyle='--')

plt.title("Training and Test Reconstruction1 Losses for Each Fold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and test losses for each fold
plt.figure(figsize=(12, 8))
for fold in range(5):
    plt.plot(fold_train_losses_recon2[fold], label=f'Fold {fold + 1} Train Reconstruction2 Loss')
    plt.plot(fold_test_losses_recon2[fold], label=f'Fold {fold + 1} Test Reconstruction2 Loss', linestyle='--')

plt.title("Training and Test Reconstruction2 Losses for Each Fold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and test losses for each fold
for fold in range(5):
    plt.figure(figsize=(12, 6))
    plt.plot(fold_train_losses[fold], label=f'Fold {fold + 1} Train Loss', color='blue')
    plt.plot(fold_test_losses[fold], label=f'Fold {fold + 1} Test Loss', color='orange', linestyle='--')
    plt.title(f"Training and Test Losses for Fold {fold + 1}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot training and test losses for each fold
for fold in range(5):
    plt.figure(figsize=(12, 6))
    plt.plot(fold_train_losses_recon1[fold], label=f'Fold {fold + 1} Train Reconstruction1 Loss', color='blue')
    plt.plot(fold_test_losses_recon1[fold], label=f'Fold {fold + 1} Test Reconstruction1 Loss', color='orange', linestyle='--')
    plt.title(f"Training and Test Reconstruction1 Losses for Fold {fold + 1}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot training and test losses for each fold
for fold in range(5):
    plt.figure(figsize=(12, 6))
    plt.plot(fold_train_losses_recon2[fold], label=f'Fold {fold + 1} Train Reconstruction2 Loss', color='blue')
    plt.plot(fold_test_losses_recon2[fold], label=f'Fold {fold + 1} Test Reconstruction2 Loss', color='orange', linestyle='--')
    plt.title(f"Training and Test Reconstruction2 Losses for Fold {fold + 1}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Healthy Control data (fixed for evaluating)
X1_valmod_hc_tensor = torch.tensor(X1_hc, dtype=torch.float32)
X2_valmod_hc_tensor = torch.tensor(X2_hc, dtype=torch.float32)

# Patient data (fixed for evaluating)
X1_valmod_ad_tensor = torch.tensor(X1_ad, dtype=torch.float32)
X2_valmod_ad_tensor = torch.tensor(X2_ad, dtype=torch.float32)

"""# Using MAE to show differences"""

model.eval()
with torch.no_grad():
    # Reconstruct
    recon_hc_m1, recon_hc_m2, _, _ = model(X1_valmod_hc_tensor, X2_valmod_hc_tensor, demo_hc_tensor)
    recon_ad_m1, recon_ad_m2, _, _ = model(X1_valmod_ad_tensor, X2_valmod_ad_tensor, demo_ad_tensor)

    # 🔹 Mean Absolute Error (MAE)
    M1_diff_hc = F.l1_loss(X1_valmod_hc_tensor, recon_hc_m1, reduction='none')
    M2_diff_hc = F.l1_loss(X2_valmod_hc_tensor, recon_hc_m2, reduction='none')
    M1_diff_ad = F.l1_loss(X1_valmod_ad_tensor, recon_ad_m1, reduction='none')
    M2_diff_ad = F.l1_loss(X2_valmod_ad_tensor, recon_ad_m2, reduction='none')

"""# Obtaining shapes and other properties of Data"""

print(f"Shape of X1_valmod_hc_tensor: {X1_valmod_hc_tensor.shape}")
print(f"Shape of X2_valmod_hc_tensor: {X2_valmod_hc_tensor.shape}")
print(f"Shape of X1_valmod_ad_tensor: {X1_valmod_ad_tensor.shape}")
print(f"Shape of X2_valmod_ad_tensor: {X2_valmod_ad_tensor.shape}")

print(f"Shape of recon_hc_m1: {recon_hc_m1.shape}")
print(f"Shape of recon_hc_m2: {recon_hc_m2.shape}")
print(f"Shape of recon_ad_m1: {recon_ad_m1.shape}")
print(f"Shape of recon_ad_m2: {recon_ad_m2.shape}")

print(f"Shape of M1_diff_hc: {M1_diff_hc.shape}")
print(f"Shape of M2_diff_hc: {M2_diff_hc.shape}")
print(f"Shape of M1_diff_ad: {M1_diff_ad.shape}")
print(f"Shape of M2_diff_ad: {M2_diff_ad.shape}")

print(X1_valmod_hc_tensor)

print(X2_valmod_hc_tensor)

print(X1_valmod_ad_tensor)

print(X2_valmod_ad_tensor)

print(recon_hc_m1)

print(recon_hc_m2)

print(recon_hc_m1)

print(recon_hc_m2)

print(M1_diff_hc)

print(M2_diff_hc)

print(M1_diff_ad)

print(M2_diff_ad)

print(M1_diff_hc[:, :8])

print(M2_diff_hc[:, :8])

print(M1_diff_ad[:, :8])

print(M2_diff_ad[:, :8])

# We want to plot features 0 through 7
# selected_features = [0, 1, 2, 3, 4, 5, 6, 7]  # Indices of features to plot
selected_features = np.arange(116)

def plot_histograms(data1, data2, data3, data4, title1, title2, title3, title4, feature_indices):
    """
    Plots individual histograms for each dataset and an overlayed histogram for each selected feature.

    Parameters:
    - data1, data2, data3, data4: Numpy arrays of shape (N, 116).
    - feature_indices: List of feature indices to plot (max 8 features).
    """
    num_features = len(feature_indices)

    titles = [title1, title2, title3, title4]
    data_sets = [data1, data2, data3, data4]
    colors = ["blue", "cyan", "green", "orange"]

    fig, axes = plt.subplots(num_features, 5, figsize=(25, num_features * 5))  # 5 columns per feature
    fig.suptitle("Histogram Comparisons with Overlay for Each Feature", fontsize=16)

    for row_idx, feature in enumerate(feature_indices):
        # Plot individual histograms
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.hist(data_sets[col_idx][:, feature], bins=20, alpha=0.7, color=colors[col_idx])
            ax.set_title(f"Feature {feature+1} - {titles[col_idx]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        # Plot overlayed histogram in the last column
        ax = axes[row_idx, 4]
        for data, color, label in zip(data_sets, colors, titles):
            ax.hist(data[:, feature], bins=20, alpha=0.5, color=color, label=label)
        ax.set_title(f"Feature {feature+1} - Overlay")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

title1 = "PET Input Data (Healthy Control)"
title2 = "PET Reconstruction Data (Healthy Control)"
title3 = "MRI Input Data (Healthy Control)"
title4 = "MRI Reconstruction Data (Healthy Control)"
plot_histograms(X1_valmod_hc_tensor, recon_hc_m1, X2_valmod_hc_tensor, recon_hc_m2, title1, title2, title3, title4, selected_features)

title1 = "PET Input Data (Alzheimer Disease)"
title2 = "PET Reconstruction Data (Alzheimer Disease)"
title3 = "MRI Input Data (Alzheimer Disease)"
title4 = "MRI Reconstruction Data (Alzheimer Disease)"
plot_histograms(X1_valmod_ad_tensor, recon_ad_m1, X2_valmod_ad_tensor, recon_ad_m2, title1, title2, title3, title4, selected_features)

title1 = "PET differenes between Input and Recon (Healthy Control)"
title2 = "MRI Differences between Input and Recon (Healthy Control)"
title3 = "PET Differences between Input and Recon (Alzheimer Disease)"
title4 = "MRI Differences between Input and Recon (Alzheimer Disease)"
plot_histograms(M1_diff_hc, M2_diff_hc, M1_diff_ad, M2_diff_ad, title1, title2, title3, title4, selected_features)

# Function to plot distributions for 8 features

def plot_kde_distributions(data1, data2, data3, data4, title1, title2, title3, title4, feature_indices):
    """
    Plots KDE distributions for individual datasets and an overlayed KDE for each selected feature.

    Parameters:
    - data1, data2, data3, data4: Numpy arrays of shape (N, 116).
    - feature_indices: List of feature indices to plot (max 8 features).
    """
    num_features = len(feature_indices)

    titles = [title1, title2, title3, title4]
    data_sets = [data1, data2, data3, data4]
    colors = ["blue", "cyan", "green", "orange"]

    fig, axes = plt.subplots(num_features, 5, figsize=(25, num_features * 5))  # 5 columns per feature
    fig.suptitle("KDE Distributions with Overlay for Each Feature", fontsize=16)

    for row_idx, feature in enumerate(feature_indices):
        # Plot individual KDE distributions
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            sns.kdeplot(data_sets[col_idx][:, feature], fill=True, color=colors[col_idx], alpha=0.6, ax=ax)
            ax.set_title(f"Feature {feature+1} - {titles[col_idx]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")

        # Plot overlayed KDE distribution in the last column
        ax = axes[row_idx, 4]
        for data, color, label in zip(data_sets, colors, titles):
            sns.kdeplot(data[:, feature], fill=True, color=color, alpha=0.4, label=label, ax=ax)
        ax.set_title(f"Feature {feature+1} - Overlay")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


title1 = "PET Input data (Healthy Control)"
title2 = "PET Reconstruction data (Healthy Control)"
plot_compare_histograms(X1_valmod_hc_tensor, recon_hc_m1, title1, title2, selected_features)

title1 = "MRI Input data (Healthy Control)"
title2 = "MRI Reconstruction data (Healthy Control)"
plot_compare_histograms(X2_valmod_hc_tensor, recon_hc_m2, title1, title2, selected_features)

title1 = "PET Input data (Alzheimer Disease)"
title2 = "PET Reconstruction data (Alzheimer disease)"
plot_compare_histograms(X1_valmod_ad_tensor, recon_ad_m1, title1, title2, selected_features)

title1 = "MRI Input data (Alzheimer Disease)"
title2 = "MRI Reconstruction data (Alzheimer)"
plot_compare_histograms(X2_valmod_ad_tensor, recon_ad_m2, title1, title2, selected_features)

def plot_dotplot_with_crossbar(data, title):
    """
    Create a dot plot with crossbars showing mean and standard deviation for each feature.

    Args:
        data: 2D numpy array with shape (samples, features), where features are plotted on the x-axis.
        title: Title for the plot.
    """
    # Calculate mean and standard deviation for each feature (column)
    feature_means = np.mean(data, axis=0)
    feature_stds = np.std(data, axis=0)
    num_features = data.shape[1]

    # Create the plot
    plt.figure(figsize=(16, 6))
    x_positions = np.arange(num_features)  # Feature indices

    # Scatter plot for individual data points
    for i in range(data.shape[0]):
        plt.scatter(x_positions, data[i], color='gray', alpha=0.5, s=10, label='_nolegend_')

    # Add crossbars (mean and std)
    plt.errorbar(
        x_positions, feature_means, yerr=feature_stds, fmt='o', color='blue', ecolor='red',
        elinewidth=1, capsize=3, label='Mean ± STD'
    )

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.xticks(ticks=x_positions, labels=np.arange(1, num_features + 1), rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.show()

plot_dotplot_with_crossbar(M1_diff_hc.cpu().numpy(), "Dot Plot with Crossbar: Difference between Original data and reconstructed data in Modality 1 (Healthy Control data)")

plot_dotplot_with_crossbar(M2_diff_hc.cpu().numpy(), "Dot Plot with Crossbar: Difference between Original data and reconstructed data in Modality 2 (Healthy Control data)")

plot_dotplot_with_crossbar(M1_diff_ad.cpu().numpy(), "Dot Plot with Crossbar: Difference between Original data and reconstructed data in Modality 1 (Patient data)")

plot_dotplot_with_crossbar(M2_diff_ad.cpu().numpy(), "Dot Plot with Crossbar: Difference between Original data and reconstructed data in Modality 2 (Patient data)")

M1_diff_hc = M1_diff_hc[:, :roi_dim]
M2_diff_hc = M2_diff_hc[:, :roi_dim]

M1_diff_ad = M1_diff_ad[:, :roi_dim]
M2_diff_ad = M2_diff_ad[:, :roi_dim]

M1_diff_hc = M1_diff_hc.cpu().numpy()
M2_diff_hc = M2_diff_hc.cpu().numpy()

M1_diff_ad = M1_diff_ad.cpu().numpy()
M2_diff_ad = M2_diff_ad.cpu().numpy()

# Function to calculate deviations
def calculate_deviations(M1_diff_hc, M1_dij_ad, M2_diff_hc, M2_dij_ad):

    """Calculate the deviations for disease patients compared to healthy controls."""

    M1_dij_hc_mean = np.mean(M1_diff_hc, axis=0)
    M1_dij_hc_std = np.std(M1_diff_hc, axis=0)

    print(f"Modality1 diff (difference between ground truth and reconstruction) Healthy Conrol data mean shape is : {M1_dij_hc_mean.shape}")
    print(f"Modality1 diff (difference between ground truth and reconstruction) Healthy Conrol data mean (per column): {M1_dij_hc_mean}")
    print(f"Modality1 diff (difference between ground truth and reconstruction) Healthy Conrol data std shape: {M1_dij_hc_std.shape}")
    print(f"Modality1 diff (difference between ground truth and reconstruction) Healthy Conrol data std (per column): {M1_dij_hc_std}")

    # Calculate Deviation for Modality 1
    M1_dev = (M1_dij_ad - M1_dij_hc_mean) / M1_dij_hc_std

    M2_dij_hc_mean = np.mean(M2_diff_hc, axis=0)
    M2_dij_hc_std = np.std(M2_diff_hc, axis=0)

    print(f"Modality2 diff (difference between ground truth and reconstruction) Healthy Conrol data mean shape is : {M2_dij_hc_mean.shape}")
    print(f"Modality2 diff (difference between ground truth and reconstruction) Healthy Conrol data mean (per column): {M2_dij_hc_mean}")
    print(f"Modality2 diff (difference between ground truth and reconstruction) Healthy Conrol data std shape: {M2_dij_hc_std.shape}")
    print(f"Modality2 diff (difference between ground truth and reconstruction) Healthy Conrol data std (per column): {M2_dij_hc_std}")

    # Calculate Deviation for Modality 2
    M2_dev = (M2_dij_ad - M2_dij_hc_mean) / M2_dij_hc_std

    return M1_dev, M2_dev

# Calculate deviations
M1_dev, M2_dev = calculate_deviations(M1_diff_hc, M1_diff_ad, M2_diff_hc, M2_diff_ad)

print(M1_dev)

print(M2_dev)

print(M1_dev.shape)
print(M2_dev.shape)

def plot_dev_histograms(M1_dev, M2_dev, feature_indices):
    """
    Plots individual histograms for each dataset and an overlayed histogram for each selected feature.

    Parameters:
    - M1_dev, M2_dev: Numpy arrays of shape (N, 116).
    - feature_indices: List of feature indices to plot (max 8 features).
    """
    num_features = len(feature_indices)


    titles = ["PET Deviation", "MRI Deviation"]
    data_sets = [M1_dev, M2_dev]
    colors = ["blue", "cyan"]

    fig, axes = plt.subplots(num_features, 3, figsize=(25, num_features * 3))  # 3 columns per feature
    fig.suptitle("Histogram Comparisons with Overlay for Each Feature", fontsize=16)

    for row_idx, feature in enumerate(feature_indices):
        # Plot individual histograms
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            ax.hist(data_sets[col_idx][:, feature], bins=20, alpha=0.7, color=colors[col_idx])
            ax.set_title(f"Feature {feature+1} - {titles[col_idx]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        # Plot overlayed histogram in the last column
        ax = axes[row_idx, 2]
        for data, color, label in zip(data_sets, colors, titles):
            ax.hist(data[:, feature], bins=20, alpha=0.5, color=color, label=label)
        ax.set_title(f"Feature {feature+1} - Overlay")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_dev_histograms(M1_dev, M2_dev, selected_features)

def identify_significant_deviations(M1_dev, M2_dev, alpha=0.05):
    """
    Identify brain regions with significant deviations based on Z-scores for modalities 1 and 2.

    Args:
        M1_dev: 2D array of shape (patients, regions) for Z-scores in modality 1.
        M2_dev: 2D array of shape (patients, regions) for Z-scores in modality 2.
        alpha: Significance threshold for identifying significant deviations (default=0.05).

    Returns:
        significant_regions: A dictionary with patient indices as keys, and for each patient, a list of regions with significant deviations in M1 and M2.
    """
    # Calculate p-values from Z-scores for both modalities
    M1_p_values = 2 * norm.sf(np.abs(M1_dev))  # Two-tailed test
    M2_p_values = 2 * norm.sf(np.abs(M2_dev))  # Two-tailed test

    significant_regions = {}

    for patient_idx in range(M1_dev.shape[0]):
        # Find regions with significant deviations in M1 and M2
        significant_M1 = np.where(M1_p_values[patient_idx] < alpha)[0]
        significant_M2 = np.where(M2_p_values[patient_idx] < alpha)[0]

        # Add to the result if there are significant deviations
        if len(significant_M1) > 0 or len(significant_M2) > 0:
            significant_regions[patient_idx] = {
                "M1_significant_regions": significant_M1.tolist(),
                "M2_significant_regions": significant_M2.tolist(),
            }

    # Print the results
    for patient, regions in significant_regions.items():
        print(f"Patient {patient + 1}:")
        if len(regions["M1_significant_regions"]) > 0:
            print(f"  Significant regions in Modality 1: {regions['M1_significant_regions']}")
        if len(regions["M2_significant_regions"]) > 0:
            print(f"  Significant regions in Modality 2: {regions['M2_significant_regions']}")

    return significant_regions

# Shape: (140 patients, 116 regions)
significant_regions = identify_significant_deviations(M1_dev, M2_dev, alpha=0.05)

def plot_significant_z_scores(M1_dev, M2_dev):
    """
    Plot a dot plot for Z-scores of all patients (y-axis) across features (x-axis),
    with dots colored red for significant p-values (<0.05) and blue otherwise.

    Args:
        M1_dev: 2D array (140 patients, 116 regions) of Z-scores in modality 1.
        M2_dev: 2D array (140 patients, 116 regions) of Z-scores in modality 2.
    """
    num_patients, num_features = M1_dev.shape

    # Calculate p-values for Z-scores
    p_values_m1 = 2 * (1 - norm.cdf(np.abs(M1_dev)))  # Two-tailed test
    p_values_m2 = 2 * (1 - norm.cdf(np.abs(M2_dev)))  # Two-tailed test

    # Create a dot plot for Modality 1
    plt.figure(figsize=(20, 8))
    for patient_idx in range(num_patients):
        for feature_idx in range(num_features):
            color = 'red' if p_values_m1[patient_idx, feature_idx] < 0.05 else 'black'
            plt.scatter(feature_idx, patient_idx, color=color, s=10, alpha=0.7)

    plt.title("Dot Plot of Z-Scores for Modality 1", fontsize=16)
    plt.xlabel("Features (Brain Regions)", fontsize=14)
    plt.ylabel("Patients", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

    # Create a dot plot for Modality 2
    plt.figure(figsize=(20, 8))
    for patient_idx in range(num_patients):
        for feature_idx in range(num_features):
            color = 'red' if p_values_m2[patient_idx, feature_idx] < 0.05 else 'black'
            plt.scatter(feature_idx, patient_idx, color=color, s=10, alpha=0.7)

    plt.title("Dot Plot of Z-Scores for Modality 2", fontsize=16)
    plt.xlabel("Features (Brain Regions)", fontsize=14)
    plt.ylabel("Patients", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

plot_significant_z_scores(M1_dev, M2_dev)

def plot_z_scores(M1_dev, M2_dev):
    """
    Plot Z-scores for all regions for each patient in two modalities (M1 and M2).

    Args:
        M1_dev: 2D array (139 patients, 116 regions) of Z-scores in modality 1.
        M2_dev: 2D array (139 patients, 116 regions) of Z-scores in modality 2.
    """
    num_patients, num_regions = M1_dev.shape

    for patient_idx in range(num_patients):
        plt.figure(figsize=(12, 6))
        plt.title(f"Z-Scores for Patient {patient_idx + 1}", fontsize=16)
        plt.xlabel("Region Index (Brain Regions)", fontsize=14)
        plt.ylabel("Z-Score", fontsize=14)

        # Plot Z-scores for Modality 1
        plt.plot(range(num_regions), M1_dev[patient_idx], label="Modality 1", marker='o', linestyle='-', alpha=0.7)

        # Plot Z-scores for Modality 2
        plt.plot(range(num_regions), M2_dev[patient_idx], label="Modality 2", marker='s', linestyle='--', alpha=0.7)

        # Add legend
        plt.legend(loc="upper right", fontsize=12)

        # Add grid for readability
        plt.grid(alpha=0.4)

        # Save or show the figure
        plt.tight_layout()
        plt.show()

plot_z_scores(M1_dev, M2_dev)