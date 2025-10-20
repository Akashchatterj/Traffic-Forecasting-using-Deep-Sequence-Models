import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyproj import Transformer
from geopy.distance import geodesic
from itertools import product
#from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#from tensorflow.keras.models import Sequential,load_model
#from tensorflow.keras.regularizers import l2
import warnings
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.nn.functional as F
import math
#from keras.callbacks import EarlyStopping
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import LSTM, Dense, Input,Dropout,RepeatVector,BatchNormalization
#from tensorflow.keras.initializers import GlorotUniform
#from tensorflow.keras.optimizers import Adam,RMSprop, SGD,Adagrad
#from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# Define the transformer from CA State Plane III NAD83 (EPSG:2227) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:2227", "EPSG:4326", always_xy=True)
print(f"transformer type: {type(transformer)}")

def calculate_gaps(data):
    # Initialize gap columns in DataFrame
    for i in range(1, 7):
        data[f'g{i}'] = np.nan

    for vehicle_id in data['Vehicle_ID'].unique():
        veh = data[data['Vehicle_ID'] == vehicle_id]
        if veh.empty:
            continue

        veh_pos_x = veh['Local_X'].values[0]
        veh_pos_y = veh['Local_Y'].values[0]
        veh_length = veh['v_Length'].values[0]

        # Calculate same-lane gaps
        same_lane = data[data['Lane_ID'] == veh['Lane_ID'].values[0]]
        leading_vehicles = same_lane[same_lane['Local_Y'] > veh_pos_y].sort_values(by='Local_Y')
        following_vehicles = same_lane[same_lane['Local_Y'] < veh_pos_y].sort_values(by='Local_Y', ascending=False)

        # Calculate leading vehicle gap
        if not leading_vehicles.empty:
            lead_veh = leading_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g1'] = lead_veh['Local_Y'] - veh_pos_y

        # Calculate following vehicle gap
        if not following_vehicles.empty:
            follow_veh = following_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g2'] = veh_pos_y - follow_veh['Local_Y']

        # Left lane leading and following vehicles
        left_lane = data[data['Lane_ID'] == veh['Lane_ID'].values[0] - 1]
        left_leading_vehicles = left_lane[left_lane['Local_Y'] > veh_pos_y].sort_values(by='Local_Y')
        left_following_vehicles = left_lane[left_lane['Local_Y'] < veh_pos_y].sort_values(by='Local_Y', ascending=False)

        # Left lane leading vehicle gap
        if not leading_vehicles.empty and not left_leading_vehicles.empty:
            left_lead_veh = left_leading_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g3'] = np.sqrt((leading_vehicles.iloc[0]['Local_X'] - left_lead_veh['Local_X'])**2 + (lead_veh['Local_Y'] - veh_pos_y)**2)

        # Left lane following vehicle gap
        if not following_vehicles.empty and not left_following_vehicles.empty:
            left_follow_veh = left_following_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g4'] = np.sqrt((following_vehicles.iloc[0]['Local_X'] - left_follow_veh['Local_X'])**2 + (veh_pos_y - follow_veh['Local_Y'])**2)

        # Right lane leading and following vehicles
        right_lane = data[data['Lane_ID'] == veh['Lane_ID'].values[0] + 1]
        right_leading_vehicles = right_lane[right_lane['Local_Y'] > veh_pos_y].sort_values(by='Local_Y')
        right_following_vehicles = right_lane[right_lane['Local_Y'] < veh_pos_y].sort_values(by='Local_Y', ascending=False)

        # Right lane leading vehicle gap
        if not leading_vehicles.empty and not right_leading_vehicles.empty:
            right_lead_veh = right_leading_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g5'] = np.sqrt((leading_vehicles.iloc[0]['Local_X'] - right_lead_veh['Local_X'])**2 + (lead_veh['Local_Y'] - veh_pos_y)**2)

        # Right lane following vehicle gap
        if not following_vehicles.empty and not right_following_vehicles.empty:
            right_follow_veh = right_following_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g6'] = np.sqrt((following_vehicles.iloc[0]['Local_X'] - right_follow_veh['Local_X'])**2 + (veh_pos_y - follow_veh['Local_Y'])**2)

    return data

def calculate_gaps_new(data):
    data['g1'] = np.nan
    data['g2'] = np.nan
    data['g3'] = np.nan
    data['g4'] = np.nan
    data['g5'] = np.nan
    data['g6'] = np.nan

    for vehicle_id in data['Vehicle_ID'].unique():
        veh = data[data['Vehicle_ID'] == vehicle_id]
        if veh.empty:
            continue

        veh_pos_y = veh['Local_Y'].values[0]
        veh_length = veh['v_Length'].values[0]

        # Same lane leading and following vehicles
        same_lane = data[data['Lane_ID'] == veh['Lane_ID'].values[0]]
        leading_vehicles = same_lane[same_lane['Local_Y'] > veh_pos_y].sort_values(by='Local_Y')
        following_vehicles = same_lane[same_lane['Local_Y'] < veh_pos_y].sort_values(by='Local_Y', ascending=False)

        if not leading_vehicles.empty:
            lead_veh = leading_vehicles.iloc[0]
            lead_veh_length = lead_veh['v_Length']
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g1'] = lead_veh['Local_Y'] - lead_veh_length - veh_pos_y

        if not following_vehicles.empty:
            follow_veh = following_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g2'] = veh_pos_y - veh_length - follow_veh['Local_Y']

        # Left adjacent lane leading and following vehicles
        left_lane = data[data['Lane_ID'] == veh['Lane_ID'].values[0] - 1]
        left_leading_vehicles = left_lane[left_lane['Local_Y'] > veh_pos_y].sort_values(by='Local_Y')
        left_following_vehicles = left_lane[left_lane['Local_Y'] < veh_pos_y].sort_values(by='Local_Y', ascending=False)

        if not left_leading_vehicles.empty:
            left_lead_veh = left_leading_vehicles.iloc[0]
            left_lead_veh_length = left_lead_veh['v_Length']
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g3'] = left_lead_veh['Local_Y'] - left_lead_veh_length - veh_pos_y

        if not left_following_vehicles.empty:
            left_follow_veh = left_following_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g4'] = veh_pos_y - veh_length - left_follow_veh['Local_Y']

        # Right adjacent lane leading and following vehicles
        right_lane = data[data['Lane_ID'] == veh['Lane_ID'].values[0] + 1]
        right_leading_vehicles = right_lane[right_lane['Local_Y'] > veh_pos_y].sort_values(by='Local_Y')
        right_following_vehicles = right_lane[right_lane['Local_Y'] < veh_pos_y].sort_values(by='Local_Y', ascending=False)

        if not right_leading_vehicles.empty:
            right_lead_veh = right_leading_vehicles.iloc[0]
            right_lead_veh_length = right_lead_veh['v_Length']
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g5'] = right_lead_veh['Local_Y'] - right_lead_veh_length - veh_pos_y

        if not right_following_vehicles.empty:
            right_follow_veh = right_following_vehicles.iloc[0]
            data.loc[data['Vehicle_ID'] == vehicle_id, 'g6'] = veh_pos_y - veh_length - right_follow_veh['Local_Y']

    return data

# Define preprocessing function
def preprocess_data(data):

    # Ensure Global_Time is in datetime format
    data['Global_Time'] = pd.to_datetime(data['Global_Time'], unit='ms')
#     time_diffs = data['Global_Time'].diff().dropna()
#     print(time_diffs.value_counts())

    # Ensure data is sorted by time or frame
    data = data.sort_values(by='Global_Time').copy()

    # Remove unwanted lanes
    data = data[~data['Lane_ID'].isin([6, 7, 8])].copy()

    # Convert units (feet to meters) where necessary
    columns_to_convert = ['Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_Length', 'Space_Hdwy']
    for column in columns_to_convert:
        if data[column].dtype == 'object':
            data[column] = data[column].str.replace(',', '').astype(float) * 0.3048
        else:
            data[column] = data[column].astype(float) * 0.3048

    # Convert other columns to numeric
    feature_columns = ['Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_Length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID', 'Space_Hdwy', 'Time_Hdwy']
    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Drop rows with any NaNs in feature columns and target
    data.dropna(subset=feature_columns + ['v_Vel'], inplace=True)

    # Apply gap calculations (assuming `calculate_gaps` function is defined)
    data = calculate_gaps_new(data)
    data.dropna(inplace=True)

    return data

def calculate_traffic_density(data):
    traffic_density = []
    for timestamp in data['Global_Time'].unique():
        frame_data = data[data['Global_Time'] == timestamp]
        density_dict = frame_data.groupby('Lane_ID').size().to_dict()
        traffic_density.extend([density_dict.get(lane, 0) for lane in frame_data['Lane_ID']])
    data['Traffic_Density'] = traffic_density
    return data

def calculate_speed_reduction(data):
    max_speed_per_lane = data.groupby('Lane_ID')['v_Vel'].max().to_dict()
    data['Speed_Reduction'] = data.apply(lambda row: row['v_Vel'] / max_speed_per_lane.get(row['Lane_ID'], 1), axis=1)
    return data

def calculate_jam_factor(data):
    data['JF'] = 1 - (data['Speed_Reduction'] / (data['Traffic_Density'] + 1))
    return data

def compute_haversine_loss(y_true, y_pred):
    """
    Compute Haversine loss between actual and predicted coordinates.
    """
    batch_size = y_true.shape[0]

    # Get the true and predicted coordinates from tensors.
    # Note: transformer.transform returns (x, y) which are (lon, lat) because always_xy=True.
    true_lon, true_lat = transformer.transform(
        y_true[:, 2].cpu().numpy(), y_true[:, 3].cpu().numpy()
    )
    pred_lon, pred_lat = transformer.transform(
        y_pred[:, 2].detach().cpu().numpy(), y_pred[:, 3].detach().cpu().numpy()
    )

    # Debug: Print first 5 values to confirm order.
    # print("True Latitudes (from transformer):", true_lat[:5])
    # print("True Longitudes (from transformer):", true_lon[:5])
    # print("Predicted Latitudes (from transformer):", pred_lat[:5])
    # print("Predicted Longitudes (from transformer):", pred_lon[:5])

    # Optionally, clip to ensure valid ranges
    true_lat = np.clip(true_lat, -90, 90)
    pred_lat = np.clip(pred_lat, -90, 90)
    true_lon = np.clip(true_lon, -180, 180)
    pred_lon = np.clip(pred_lon, -180, 180)

    # Compute geodesic distances (using (lat, lon) order)
    distances = np.array([
        geodesic((lat1, lon1), (lat2, lon2)).meters
        for lat1, lon1, lat2, lon2 in zip(true_lat, true_lon, pred_lat, pred_lon)
    ])

    # Return the mean squared distance (loss)
    return torch.tensor(np.mean(distances**2), dtype=torch.float32, device=y_true.device)


# Read the first few lines of the CSV file to inspect the structure
# data = pd.read_csv('/kaggle/input/ngsim-vehicle-trajectories/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv', header=0, delimiter=',', )
data1 = pd.read_csv('/DATA1/us-dataset/0750_0805_us101_smoothed_21_.csv', header=0, delimiter=',')
data2 = pd.read_csv('/DATA1/us-dataset/0805_0820_us101_smoothed_21_.csv', header=0, delimiter=',')
data3 = pd.read_csv('/DATA1/us-dataset/0820_0835_us101_smoothed_21_.csv', header=0, delimiter=',')

# Display the first few rows to verify the number of columns
print(data1.head())


# Print the number of columns detected
print(f"Number of columns detected: {data1.shape[1]}")
print(f"Number of rows detected: {data1.shape[0]}")

# Preprocess each dataset
data1 = preprocess_data(data1)
data2 = preprocess_data(data2)
data3 = preprocess_data(data3)

# Combine datasets
data_combined = pd.concat([data1, data2, data3])
data_combined['Lane_Change_Label'] = data_combined['Lane_ID'].diff().fillna(0).apply(lambda x: 1 if x > 0 else (2 if x < 0 else 0))
data_combined = calculate_traffic_density(data_combined)
data_combined = calculate_speed_reduction(data_combined)
data_combined = calculate_jam_factor(data_combined)

# Define final set of input features
final_feature_columns = ['Local_X', 'Lane_ID', 'v_Vel', 'v_Acc', 'v_Length', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'Traffic_Density', 'Speed_Reduction', 'JF']
target_columns = ['v_Vel', 'Local_Y', 'Global_X', 'Global_Y', 'Lane_Change_Label', 'JF']


# Prepare Data
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()
features_scaled = scaler_features.fit_transform(data_combined[final_feature_columns])
target_scaled = scaler_target.fit_transform(data_combined[target_columns])

features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
target_scaled = torch.tensor(target_scaled, dtype=torch.float32)

# Hyperparameters
n_input = 10
n_features = len(final_feature_columns)
n_output = 4
batch_size = 16
epochs = 10
hidden_dim = 128
output_size = 4
learning_rate = 0.0005
n_lstm_neurons = 10
alpha_values = [0.1, 0.2,0.3,0.4,0.5]
beta_values = [0.1, 0.2,0.3,0.4,0.5]
gamma_values = [0.1, 0.2,0.3,0.4,0.5]
delta_values = [0.1, 0.2,0.3,0.4,0.5]

# Custom loss printer class for PyTorch
class LossPrinter:
    def __init__(self):
        self.mse_losses_v_Vel = []      # List for MSE losses of v_Vel
        self.mse_losses_Local_Y = []    # List for MSE losses of Local_Y
        self.self_consistency_losses = []  # List for Self-Consistency losses
        self.classification_losses = []    # List for classification loss
        self.haversine_dist_losses = []   # List for Haversine Distance Loss
        self.congestion_losses = []  # List for congestion loss

    def log_epoch(self, mse_loss_v_Vel, mse_loss_Local_Y, self_consistency_loss, classification_loss, congestion_loss):
        # Store each loss after every epoch
        self.mse_losses_v_Vel.append(mse_loss_v_Vel.item())
        self.mse_losses_Local_Y.append(mse_loss_Local_Y.item())
        self.self_consistency_losses.append(self_consistency_loss.item())
        self.haversine_dist_losses.append(haversine_dist_loss.item())  # Store Haversine Distance Loss
        self.classification_losses.append(classification_loss.item())
        self.congestion_losses.append(congestion_loss.item())

        # Print the losses for each epoch
        print(f"MSE Loss of v_Vel = {mse_loss_v_Vel:.8f}, "
              f"MSE Loss of Local_Y = {mse_loss_Local_Y:.8f}, "
              f"Haversine Distance Loss = {haversine_dist_loss:.8f}, "
              f"Self-Consistency Loss = {self_consistency_loss:.8f}, "
              f"Classification Loss = {classification_loss:.8f}, "
              f"Congestion Loss = {congestion_loss:.8f}")
#       

#         # Reset the metrics at the end of each epoch
#         self.mse_metric_v_Vel.reset_state()
#         self.mse_metric_Local_Y.reset_state()
#         self.self_consistency_metric.reset_state()

def custom_loss(y_true, y_pred, class_true, class_pred,congestion_true, congestion_pred, alpha, beta, gamma,sigma, delta,transformer, time_interval=0.1):
    # Compute losses
    mse_loss_v_Vel = nn.MSELoss()(y_pred[:, 0], y_true[:, 0])
    mse_loss_Local_Y = nn.MSELoss()(y_pred[:, 1], y_true[:, 1])
    haversine_dist_loss = compute_haversine_loss(y_true, y_pred)  # Haversine loss for Global_X, Global_Y
    #mse_loss_Local_Y = nn.MSELoss()(y_pred[:, 1], y_true[:, 1])
    #haversine_dist_loss = haversine_loss(y_true, y_pred,transformer)
    self_consistency_loss = nn.L1Loss()(y_pred[:, 1], y_true[:, 1] + (y_pred[:, 0] * time_interval))
    classification_loss = nn.CrossEntropyLoss()(class_pred, class_true)
    congestion_loss = nn.MSELoss()(congestion_pred.squeeze(), congestion_true)    

    total_loss = mse_loss_v_Vel + (alpha * mse_loss_Local_Y) + \
                 (beta * self_consistency_loss) + (gamma * classification_loss) + \
                 (delta * congestion_loss) + (sigma * haversine_dist_loss)
    #total_loss = mse_loss_v_Vel + (alpha * mse_loss_Local_Y) + (beta * self_consistency_loss) + (gamma * classification_loss) + (delta * congestion_loss)
    return total_loss, mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
            print(f"Adjusted hidden_size to {hidden_size} to be divisible by num_heads ({num_heads}).")
        
        self.input_proj = nn.Linear(input_size, hidden_size)  # Project input to hidden size
        self.layer_norm = nn.LayerNorm(hidden_size)  # Normalize for stable training
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        # Separate fully connected layers for different outputs
        #self.fc_velocity = nn.Linear(hidden_size, 1)  # Predict v_Vel
        #self.fc_position = nn.Linear(hidden_size, 3)  # Predict Local_Y, Global_X, Global_Y
        self.fc_regression = nn.Linear(hidden_size, output_size)  # For velocity and Local_Y
        self.fc_classification = nn.Linear(hidden_size, 3)  # For lane-change classification
        self.fc_congestion = nn.Linear(hidden_size, 1)  # For congestion prediction

    def forward(self, x):
        # Ensure x has shape (batch_size, sequence_length, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence length dimension if missing
        
        x = self.input_proj(x)  # Project input features to hidden_size
        x = self.layer_norm(x)  # Normalize
        x = self.positional_encoding(x)  # Add positional encoding
        out = self.transformer_encoder(x)  # Pass through Transformer encoder

        # Take the last output token (equivalent to taking the last hidden state in LSTMs)
        #velocity_output = self.fc_velocity(out[:, -1, :])
        #position_output = self.fc_position(out[:, -1, :])  # Predict Local_Y, Global_X, Global_Y
        regression_output = self.fc_regression(out[:, -1, :])
        classification_output = F.softmax(self.fc_classification(out[:, -1, :]), dim=1)
        congestion_output = torch.sigmoid(self.fc_congestion(out[:, -1, :]))  # Sigmoid for binary classification

        return  regression_output, classification_output, congestion_output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Cross-validation with tuning
# Initialize best validation loss
best_val_loss = float('inf')
best_alpha, best_beta, best_gamma, best_delta, best_sigma = None, None, None, None, None
best_fold = None
kf = KFold(n_splits=3)

for alpha, beta, gamma, sigma, delta in product(alpha_values, beta_values, gamma_values, sigma_values, delta_values):
    print(f"Tuning for alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}, sigma={sigma}")

    fold = 1  # Reset fold counter for each hyperparameter combination
    for train_index, test_index in kf.split(features_scaled):
        print(f"Starting Cross-Validation Fold {fold}")

        # Split into training and testing sets
        features_train, features_test = features_scaled[train_index], features_scaled[test_index]
        targets_train, targets_test = target_scaled[train_index], target_scaled[test_index]

        # Further split training set into training and validation
        features_train, features_val, targets_train, targets_val = train_test_split(
            features_train, targets_train, test_size=0.2, random_state=42
        )

        # Create DataLoader for PyTorch
        train_dataset = TensorDataset(features_train, targets_train)
        val_dataset = TensorDataset(features_val, targets_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Initialize Transformer Model with correct parameters
        model = TransformerModel(n_features, hidden_size).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        loss_printer = LossPrinter()

        # Training Loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

                # Forward pass
                velocity_pred, position_pred, class_pred, congestion_pred = model(batch_features)

                # Compute loss with corrected target indexing
                combined_loss, mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss = custom_loss(
                    batch_targets[:, :4],  # Now includes v_Vel, Local_Y, Global_X, Global_Y
                    velocity_pred, position_pred,
                    batch_targets[:, 4].long(), class_pred,  # Lane change label
                    batch_targets[:, 5], congestion_pred,  # Congestion label
                    alpha, beta, gamma, sigma, delta
                )

                # Backward pass and optimization
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                train_loss += combined_loss.item()

            # Log losses
            loss_printer.log_epoch(mse_loss_v_Vel, mse_loss_Local_Y, self_consistency_loss, classification_loss, congestion_loss)

            # Validation loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    velocity_pred, position_pred, class_pred, congestion_pred = model(inputs)

                    loss, _, _, _, _, _, _ = custom_loss(
                        targets[:, :4], velocity_pred, position_pred,
                        targets[:, 4].long(), class_pred,
                        targets[:, 5], congestion_pred,
                        alpha, beta, gamma, sigma, delta
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # Check for best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_alpha, best_beta, best_gamma, best_delta, best_sigma = alpha, beta, gamma, delta, sigma
                best_fold = fold
                print(f"New best loss: {best_val_loss} with alpha={best_alpha}, beta={best_beta}, gamma={best_gamma}, delta={best_delta}, sigma={best_sigma} on fold {best_fold}")

        fold += 1

print(f"Best alpha: {best_alpha}, Best beta: {best_beta}, Best gamma={best_gamma}, Best delta={best_delta}, Best sigma={best_sigma}, Best fold: {best_fold}, Best validation loss: {best_val_loss}")

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon):
    distances = [
        geodesic((lat1, lon1), (lat2, lon2)).meters
        for lat1, lon1, lat2, lon2 in zip(actual_lat, actual_lon, pred_lat, pred_lon)
    ]
    return np.sqrt(np.mean(np.square(distances)))  # RMSE of distances

# Example
# Best hyperparameters from tuning
#best_alpha, best_beta, best_gamma, best_delta, best_sigma = 0.1, 0.1, 0.1, 0.1, 0.1
#best_fold = 2  # Train only on this fold

# Where to save the final model & loss histories
save_root     = "/home/m23air002/"  
final_ckpt_dir = os.path.join(save_root, "final_model/")
os.makedirs(final_ckpt_dir, exist_ok=True)

# Cross-validation and model training
# Loss tracking for plotting
loss_printer_callback = {
    'epochs': [],
    'total_losses': [],
    'mse_losses_v_Vel': [],
    'mse_losses_Local_Y': [],
    'mse_losses_Global_X': [],
    'mse_losses_Global_Y': [],
    'haversine_dist_losses':[],
    'self_consistency_losses': [],
    'classification_losses': [],
    'congestion_losses': []
}

# Cross-validation setup
kf = KFold(n_splits=3, shuffle=False)
fold = 1
rmse_results = []

for train_index, test_index in kf.split(features_scaled):
    # if fold != best_fold:
    #     fold += 1
    #     continue  # Skip other folds

    # print(f"Training on Best Fold {best_fold}")

    # Split data
    features_train, features_test = features_scaled[train_index], features_scaled[test_index]
    targets_train, targets_test = target_scaled[train_index], target_scaled[test_index]

    # Split into training & validation
    features_train, features_val, targets_train, targets_val = train_test_split(
        features_train, targets_train, test_size=0.2, random_state=42
    )

    # Create PyTorch Datasets
    train_dataset = TensorDataset(torch.FloatTensor(features_train), torch.FloatTensor(targets_train))
    val_dataset = TensorDataset(torch.FloatTensor(features_val), torch.FloatTensor(targets_val))
    test_dataset = TensorDataset(torch.FloatTensor(features_test), torch.FloatTensor(targets_test))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = TransformerModel(input_size=features_train.shape[1], hidden_size=hidden_dim,output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    # Initialize fold-wise storage at the start of each fold
    loss_printer_callback[fold] = {
    'epochs':[],
    'total_losses': [],
    'mse_losses_v_Vel': [],
    'mse_losses_Local_Y': [],
    'haversine_dist_losses': [],
    'self_consistency_losses': [],
    'classification_losses': [],
    'congestion_losses': []
    }

    # Training Loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        mse_v_Vel, mse_Local_Y, haversine_loss, self_consistency_loss, classification_loss, congestion_loss = 0, 0, 0, 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward Pass
            predictions, class_pred, congestion_pred = model(inputs)

            # Compute Loss
            loss, mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss = custom_loss(
                labels[:, :4], predictions,  # v_Vel, Local_Y, Global_X, Global_Y
                labels[:, 4].long(), class_pred,  # Lane Change Labels
                labels[:, 5], congestion_pred,  # Congestion Labels
                best_alpha, best_beta, best_gamma, best_sigma, best_delta,transformer
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Sum up the loss components for tracking
            mse_v_Vel += mse_loss_v_Vel.item()
            mse_Local_Y += mse_loss_Local_Y.item()
            haversine_loss += haversine_dist_loss.item()
            self_consistency_loss += self_consistency_loss.item()
            classification_loss += classification_loss.item()
            congestion_loss += congestion_loss.item()

        torch.cuda.empty_cache()  # Free memory after each epoch

        # Store Losses for Plotting
        loss_printer_callback[fold]['epochs'].append(epoch + 1)
        loss_printer_callback[fold]['total_losses'].append(epoch_loss / len(train_loader))
        loss_printer_callback[fold]['mse_losses_v_Vel'].append(mse_v_Vel / len(train_loader))
        loss_printer_callback[fold]['mse_losses_Local_Y'].append(mse_Local_Y / len(train_loader))
        loss_printer_callback[fold]['haversine_dist_losses'].append(haversine_loss / len(train_loader))
        loss_printer_callback[fold]['self_consistency_losses'].append(self_consistency_loss.detach().cpu().numpy() / len(train_loader))
        loss_printer_callback[fold]['classification_losses'].append(classification_loss.detach().cpu().numpy() / len(train_loader))
        loss_printer_callback[fold]['congestion_losses'].append(congestion_loss.detach().cpu().numpy() / len(train_loader))
        
        # Save model weights only at the last epoch for every fold
        if epoch == epochs - 1:  
            torch.save(model.state_dict(), f'Transformer_fold{fold}_epoch{epoch + 1}.pth')

    #  Model Evaluation on Test Data 
    model.eval()  # **Set model to evaluation mode**

    predictions, lane_change_preds, congestion_preds = [], [], []
    actual_v_Vel_list, actual_local_Y_list, actual_global_X_list, actual_global_Y_list = [], [], [], []
    actual_lane_change_list, actual_congestion_list, actual_lat_list, actual_lon_list,pred_lat_list,pred_lon_list = [],[],[],[],[],[]

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward Pass
            predictions_batch, class_pred, congestion_pred = model(inputs)

            # Convert actual Global_X, Global_Y to Lat/Lon
            for i in range(len(labels)):
                global_x, global_y = labels[i, 2].cpu().item(), labels[i, 3].cpu().item()
                lon, lat = transformer.transform(global_x, global_y)
                actual_lat_list.append(lat)
                actual_lon_list.append(lon)

            # Convert predicted Global_X, Global_Y to Lat/Lon
            for i in range(len(predictions_batch)):
                pred_x, pred_y = predictions_batch[i, 2].cpu().item(), predictions_batch[i, 3].cpu().item()
                pred_lon, pred_lat = transformer.transform(pred_x, pred_y)
                pred_lat_list.append(pred_lat)
                pred_lon_list.append(pred_lon)


            # Store predictions
            predictions.append(predictions_batch.cpu())
            lane_change_preds.append(class_pred.cpu().numpy())
            congestion_preds.append(congestion_pred.cpu().numpy())

            # Store actual labels
            actual_v_Vel_list.append(labels[:, 0].cpu().numpy())
            actual_local_Y_list.append(labels[:, 1].cpu().numpy())
            actual_global_X_list.append(labels[:, 2].cpu().numpy())
            actual_global_Y_list.append(labels[:, 3].cpu().numpy())
            actual_lane_change_list.append(labels[:, 4].cpu().numpy())
            actual_congestion_list.append(labels[:, 5].cpu().numpy())

    # Convert to numpy
    predictions = np.concatenate(predictions)
    lane_change_preds = np.concatenate(lane_change_preds)
    congestion_preds = np.concatenate(congestion_preds).squeeze()
    actual_v_Vel = np.concatenate(actual_v_Vel_list)
    actual_local_Y = np.concatenate(actual_local_Y_list)
    actual_lat = np.array(actual_lat_list)  # Convert latitudes to numpy array
    actual_lon = np.array(actual_lon_list)  # Convert longitudes to numpy array
    pred_lat = np.array(pred_lat_list)  # Convert predicted latitudes to numpy array
    pred_lon = np.array(pred_lon_list)  # Convert predicted longitudes to numpy array
    actual_lane_change = np.concatenate(actual_lane_change_list).astype(int)
    actual_congestion = np.concatenate(actual_congestion_list)

    # Convert lane-change probabilities to class labels
    predicted_lane_change = np.argmax(lane_change_preds, axis=1)  # Get class with highest probability
    actual_lane_change = actual_lane_change.astype(int)  # Convert actual values to integers
    predicted_lane_change = predicted_lane_change.astype(int)  # Convert predicted values to integers


    # RMSE Metrics
    rmse_v_Vel = calculate_rmse(actual_v_Vel, predictions[:, 0])
    print(f"RMSE for velocity prediction: {rmse_v_Vel}")
    rmse_local_Y = calculate_rmse(actual_local_Y, predictions[:, 1])
    print(f"RMSE for local Y prediction: {rmse_local_Y}")
    rmse_haversine = haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon)
    print(f"Haversine RMSE (meters): {rmse_haversine}")
    congestion_preds = congestion_preds.squeeze()  # Removes extra dimensions
    rmse_congestion = calculate_rmse(actual_congestion, congestion_preds)
    print(f"RMSE for congestion prediction: {rmse_congestion}")

    # Compute Lane Change Classification Metrics
    accuracy = accuracy_score(actual_lane_change, predicted_lane_change)
    precision = precision_score(actual_lane_change, predicted_lane_change, average='macro')
    recall = recall_score(actual_lane_change, predicted_lane_change, average='macro')
    f1 = f1_score(actual_lane_change, predicted_lane_change, average='macro')

    print(f"Lane Change Prediction Accuracy: {accuracy:.4f}")
    print(f"Lane Change Precision: {precision:.4f}")
    print(f"Lane Change Recall: {recall:.4f}")
    print(f"Lane Change F1 Score: {f1:.4f}")

    # Store results
    rmse_results.append({
        'fold': fold,
        'rmse_v_Vel': rmse_v_Vel,
        'rmse_local_Y': rmse_local_Y,
        'Haversine RMSE':rmse_haversine,
        'rmse_congestion': rmse_congestion,
        'lane_change_accuracy': accuracy,
        'lane_change_precision': precision,
        'lane_change_recall': recall,
        'lane_change_f1': f1,
        'actual_lat': actual_lat,  # Store actual latitudes
        'actual_lon': actual_lon,  # Store actual longitudes
        'pred_lat': pred_lat,  # Store predicted latitudes
        'pred_lon': pred_lon,   # Store predicted longitudes
        'predicted lane':predicted_lane_change
    })

    fold += 1

# Print Evaluation Metrics
print("\nSummary of RMSE Transformer across folds:")
print(f"{'Fold':<5} {'RMSE for velocity':<20} {'RMSE for Local_Y':<20} {'Haversine RMSE':<20} {'RMSE for congestion':<20}   {'lane_change_accuracy':<20}  {'pred_lat':<20}  {'pred_lon':<20}")
for result in rmse_results:
    avg_pred_lat = np.mean(result['pred_lat'])  # Take mean predicted latitude
    avg_pred_lon = np.mean(result['pred_lon'])  # Take mean predicted longitude
    #avg_pred_lane = np.mean(result['predicted_lane'])  # Average predicted lane (for summary)
    print(f"{result['fold']:<5} {result['rmse_v_Vel']:<20.5f} {result['rmse_local_Y']:<20.5f} {result['Haversine RMSE']:<20.5f} {result['rmse_congestion']:<20.5f} {result['lane_change_accuracy']:<20.5f}  {avg_pred_lat:<15.5f} {avg_pred_lon:<15.5f}")

    

# Adjust global font sizes for readability
plt.rcParams.update({
    'axes.titlesize': 18,      # Font size for plot title
    'axes.labelsize': 18,      # Font size for x and y axis labels
    'xtick.labelsize': 14,     # Font size for x axis tick labels
    'ytick.labelsize': 14,     # Font size for y axis tick labels
    'legend.fontsize': 14,     # Font size for legend labels
    'figure.figsize': (14, 10) # Adjust figure size (optional)
})

# Plotting the total loss
save_root = "/home/m23air002/"  

for fold in loss_printer_callback:
    if isinstance(fold, int):  
        save_dir = os.path.join(save_root, f"fold{fold}_Transformer/")
        os.makedirs(save_dir, exist_ok=True)  

        plt.figure(figsize=(10, 6))
        plt.plot(loss_printer_callback[fold]['total_losses'], label=f'Fold {fold}')

        plt.title(f'Total Loss for Fold {fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Set x-axis ticks to integers
        plt.xticks(ticks=range(0, len(loss_printer_callback[fold]['total_losses']), 5))

        plot_path = os.path.join(save_dir, f"total_loss_plot_fold{fold}.pdf")
        plt.savefig(plot_path)
        plt.show()

        print(f"Plot saved for Fold {fold} at: {plot_path}")

plt.figure()

for fold in loss_printer_callback:
    if isinstance(fold, int) and 'mse_losses_v_Vel' in loss_printer_callback[fold]:  
        plt.figure(figsize=(10, 6))

        # Plot losses for the current fold
        plt.plot(loss_printer_callback[fold]['mse_losses_v_Vel'], label='MSE Loss v_Vel')
        plt.plot(loss_printer_callback[fold]['mse_losses_Local_Y'], label='MSE Loss Local_Y')
        plt.plot(loss_printer_callback[fold]['haversine_dist_losses'], label='Haversine Distance Loss')
        plt.plot(loss_printer_callback[fold]['congestion_losses'], label='Congestion Loss')
        plt.plot(loss_printer_callback[fold]['classification_losses'], label='Classification Loss')
        plt.plot(loss_printer_callback[fold]['self_consistency_losses'], label='Self-Consistency Loss')

        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'MSE and Self-Consistency Losses Transformer - Fold {fold}')
        plt.legend()
        plt.grid(True)

        # Define save directory and save path
        save_dir = "/home/m23air002/"  
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"MSE_and_Self_Consistency_Losses_Fold{fold}_Transformer.pdf")

        # Save the plot
        plt.savefig(plot_path)
        plt.show()

        print(f"Plot saved for Fold {fold} at: {plot_path}")

#Ensure your predicted and actual values are NumPy arrays for consistent plotting
actual_v_Vel = np.array(actual_v_Vel)
predicted_v_Vel = np.array(predictions[:, 0])
actual_local_Y = np.array(actual_local_Y)
predicted_local_Y = np.array(predictions[:, 1])
actual_congestion = np.array(actual_congestion).reshape(-1)
predicted_congestion = np.array(congestion_preds).reshape(-1)
# # Downsample by taking every 100th sample
# downsampled_indices = np.arange(0, len(actual_v_Vel), 1000)
# actual_v_Vel_downsampled = actual_v_Vel[downsampled_indices]
# predicted_v_Vel_downsampled = predicted_v_Vel[downsampled_indices]
# actual_local_Y_downsampled = actual_local_Y[downsampled_indices]
# predicted_local_Y_downsampled = predicted_local_Y[downsampled_indices]
# actual_congestion_downsampled = actual_congestion[downsampled_indices]
# predicted_congestion_downsampled = predicted_congestion[downsampled_indices]


# (2) Take a contiguous slice of the first N = 800 samples
N = 800
av = actual_v_Vel[:N]
pv = predicted_v_Vel[:N]
al = actual_local_Y[:N]
pl = predicted_local_Y[:N]
ac = actual_congestion[:N]
pc = predicted_congestion[:N]

# (3) Define a small moving‐average function (window size = 10)
def moving_average(x, k=10):
    """Return a 1D moving average of x with window size k (centered)."""
    return np.convolve(x, np.ones(k)/k, mode='same')

# (4) Smooth each array with that moving average
av_sm = moving_average(av, k=10)
pv_sm = moving_average(pv, k=10)
al_sm = moving_average(al, k=10)
pl_sm = moving_average(pl, k=10)
ac_sm = moving_average(ac, k=10)
pc_sm = moving_average(pc, k=10)

# (5) Increase font sizes so text is legible on A4
plt.rcParams.update({
    'axes.titlesize':   20,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'figure.figsize':  (12, 6),
})

# Where to write
out_dir = "/home/m23air002"
os.makedirs(out_dir, exist_ok=True)

# 1) Plot Actual
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(av_sm, label='Actual v_Vel', color='blue', alpha=0.8, linestyle='-')
ax.set_title('Actual Vehicle Velocity (v_Vel)', fontsize=20)
ax.set_xlabel('Sample Index', fontsize=16)
ax.set_ylabel('Velocity', fontsize=16)
ax.legend(loc='lower right', fontsize=14)
plt.xlim(1,N-1)
ax.grid(True)
fig.tight_layout()

actual_path = os.path.join(out_dir, "actual_v_Vel_smoothed(Transformer).pdf")
fig.savefig(actual_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"✔ Saved actual plot to {actual_path}")

# 2) Plot Predicted
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(pv_sm, label='Predicted v_Vel', color='orange', alpha=0.8, linestyle='--')
ax.set_title('Predicted Vehicle Velocity (v_Vel)', fontsize=20)
ax.set_xlabel('Sample Index', fontsize=16)
ax.set_ylabel('Velocity', fontsize=16)
ax.legend(loc='lower right', fontsize=14)
plt.xlim(1,N-1)
ax.grid(True)
fig.tight_layout()

pred_path = os.path.join(out_dir, "predicted_v_Vel_smoothed(Transformer).pdf")
fig.savefig(pred_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"✔ Saved predicted plot to {pred_path}")

# (7) Plot “Local_Y” in its own figure
plt.figure()
plt.plot(al_sm, label='Actual Local_Y ',    color='green', alpha=0.8, linestyle='-')
plt.plot(pl_sm, label='Predicted Local_Y ', color='red',   alpha=0.8, linestyle='--')
plt.title('Predicted vs Actual Local_Y (Position)', fontsize=20)
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('Local_Y',    fontsize=16)
plt.legend(loc='lower right')
plt.xlim(1,N-1)
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/m23air002/predicted_vs_actual_Local_Y_smoothed(Transformer).pdf')
plt.show()

# (8) Plot “Congestion” in its own figure
plt.figure()
plt.plot(ac_sm, label='Actual Congestion',    color='green', alpha=0.8, linestyle='-')
plt.plot(pc_sm, label='Predicted Congestion', color='red',   alpha=0.8, linestyle='--')
plt.title('Predicted vs Actual Congestion', fontsize=20)
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('Congestion Metric',   fontsize=16)
plt.legend(loc='lower right')
plt.xlim(1,N-1)
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/m23air002/predicted_vs_actual_Congestion_smoothed(Transformer).pdf')
plt.show()

# Adjust layout for better visibility
#plt.tight_layout()

# Save the plots
#plt.savefig('/home/m23air002/predictions_vs_actuals(Transformer).png')

# Show the plots
#plt.show()

# # Assuming rmse_results is a list of tuples or lists like [(fold, rmse_v_Vel, rmse_local_Y), ...]
# rmse_results_df = pd.DataFrame(rmse_results, columns=['Fold', 'RMSE for velocity', 'RMSE for Local_Y','RMSE for Congestion', 'lane_change_accuracy','lane_change_precision', 'lane_change_recall', 'lane_change_f1','actual_lat','actual_lon','pred_lat','pred_lon','predicted_lane' ])

# # Save the DataFrame to a CSV file
# rmse_results_df.to_csv('Transformer_results.csv', index=False)

