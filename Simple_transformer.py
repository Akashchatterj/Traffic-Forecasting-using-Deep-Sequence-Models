import torch
import logging
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

logging.basicConfig(filename = 'transformer.log', filemode='w', format= '%(message)s', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(device)

# Define the transformer from CA State Plane III NAD83 (EPSG:2227) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:2227", "EPSG:4326", always_xy=True)
logging.info(f"transformer type: {type(transformer)}")

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
#     logging.info(time_diffs.value_counts())

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
    # logging.info("True Latitudes (from transformer):", true_lat[:5])
    # logging.info("True Longitudes (from transformer):", true_lon[:5])
    # logging.info("Predicted Latitudes (from transformer):", pred_lat[:5])
    # logging.info("Predicted Longitudes (from transformer):", pred_lon[:5])

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
data1 = pd.read_csv('/DATA2/usdataset/0750_0805_us101_smoothed_21_.csv', header=0, delimiter=',')
data2 = pd.read_csv('/DATA2/usdataset/0805_0820_us101_smoothed_21_.csv', header=0, delimiter=',')
data3 = pd.read_csv('/DATA2/usdataset/0820_0835_us101_smoothed_21_.csv', header=0, delimiter=',')

# Display the first few rows to verify the number of columns
logging.info(data1.head())


# Print the number of columns detected
logging.info(f"Number of columns detected: {data1.shape[1]}")
logging.info(f"Number of rows detected: {data1.shape[0]}")

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
final_feature_columns = ['Local_X', 'Lane_ID', 'v_Vel', 'v_Acc', 'v_Length', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6']
target_columns = ['v_Vel', 'Local_Y', 'Global_X', 'Global_Y', 'Lane_Change_Label', 'JF']

# Prepare Data
scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_target = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler_features.fit_transform(data_combined[final_feature_columns])
target_scaled = scaler_target.fit_transform(data_combined[target_columns])

features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
target_scaled = torch.tensor(target_scaled, dtype=torch.float32)

# Hyperparameters
n_input = 10
n_features = len(final_feature_columns)
n_output = len(target_columns)
batch_size = 16
epochs = 50
# It controls the internal feature size of the transformer.
d_model = 64   # d_model: dimensionality of the token embeddings and transformer layers               
n_heads = 4             # Number of attention heads
n_layers = 2            # Number of transformer encoder layers
hidden_dim = 128
output_size = 4
dropout = 0.1 
learning_rate = 0.00005
n_lstm_neurons = 10
alpha_values = [0.1, 0.2]
beta_values = [0.1, 0.5]
gamma_values = [0.1,0.5]
delta_values = [0.1,0.5]

# Custom loss printer class for PyTorch
# Custom loss printer class for PyTorch
class LossPrinter:
    def __init__(self):
        self.losses = []      # List for MSE loss

    def log_epoch(self, mse_loss_v_Vel, mse_loss_Local_Y, self_consistency_loss):
        # Store each loss after every epoch
        self.losses.append(total_loss.item())
        # Print the losses for each epoch
        logging.info(f"  Total MSE Loss = {total_loss:.8f}")

def custom_loss(y_true, y_pred):
    """
    Simple MSE loss between predictions and targets.
    """
    return nn.MSELoss()(y_pred, y_true)

# --- Model Definition ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        # take last time step
        out = self.fc_out(x[:, -1, :])
        return out

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon):
    distances = [
        geodesic((lat1, lon1), (lat2, lon2)).meters
        for lat1, lon1, lat2, lon2 in zip(actual_lat, actual_lon, pred_lat, pred_lon)
    ]
    return np.sqrt(np.mean(np.square(distances)))  # RMSE of distances

# Generate input sequences for LSTM (samples, timesteps, features)
def create_sequences(features, targets, n_input):
    sequences = []
    labels = []
    for i in range(len(features) - n_input):
        seq = features[i:i+n_input]
        label = targets[i+n_input]
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.stack(labels)

# Create sequences from scaled data
#X, y = create_sequences(features_scaled, target_scaled, n_input)


# --- GRID SEARCH SETUP ---
n_output = len(target_columns) 
# param_grid = {
#     'batch_size': [8, 16],           # e.g., edit as needed
#     'learning_rate': [1e-3, 1e-4],
#     'd_model': [32, 64],
#     'n_heads': [2, 4],
#     'n_layers': [1, 2],
#     'n_input': [5, 10],              # sequence/window length
#     'epochs': [10],                  # reduce for testing if needed
# }

# hyperparam_combinations = list(product(*param_grid.values()))
# param_keys = list(param_grid.keys())

all_results = []
loss_printer_callback = {}
# for combo in hyperparam_combinations:
#     current_params = dict(zip(param_keys, combo))
#     logging.info(f"\n🔍 Testing config: {current_params}")

#     # --- Assign params ---
#     batch_size = current_params['batch_size']
#     learning_rate = current_params['learning_rate']
#     d_model = current_params['d_model']
#     n_heads = current_params['n_heads']
#     n_layers = current_params['n_layers']
#     n_input = current_params['n_input']
#     epochs = current_params['epochs']
for i in range(1):
    # --- Create sequences with current n_input ---
    X, y = create_sequences(features_scaled, target_scaled, n_input)

    # --- Cross-Validation & Training Loop ---

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_results = []
    fold = 1

    # Define target groups for metrics
    geo_targets = ['Global_X', 'Global_Y']
    # Include congestion along with standard targets
    standard_targets = ['v_Vel', 'Local_Y', 'JF']

    # Ensure X, y, scaler_target, transformer, target_columns are defined
    # X: shape (samples, seq_len, features)
    # y: shape (samples, targets), target_columns should include 'Congestion'

    for train_index, test_index in kf.split(X):
        logging.info(f"\nStarting Cross-Validation Fold {fold}")
        loss_printer_callback[fold] = {'epoch_losses': []}

        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        # Initialize Transformer model
        model = TransformerModel(
            input_dim=n_features,
            d_model=d_model,
            output_dim=n_output,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training epochs
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = custom_loss(labels, outputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
            loss_printer_callback[fold]['epoch_losses'].append(avg_loss)

        # Evaluation on test set
        model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
                actuals.append(labels.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        # Inverse scaling
        preds_orig = scaler_target.inverse_transform(predictions)
        actuals_orig = scaler_target.inverse_transform(actuals)

        # Geospatial conversion
        actual_x, actual_y = actuals_orig[:, 2], actuals_orig[:, 3]
        pred_x, pred_y = preds_orig[:, 2], preds_orig[:, 3]
        actual_lon, actual_lat = transformer.transform(actual_x, actual_y)
        pred_lon, pred_lat = transformer.transform(pred_x, pred_y)

        # Compute RMSE per target
        rmse_all = {}
        haversine_done = False
        for idx, col in enumerate(target_columns):
            if col in geo_targets and not haversine_done:
                rmse_all['Haversine_RMSE'] = haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon)
                haversine_done = True
            else:
                # Computes RMSE for standard targets including Congestion
                rmse_all[col] = calculate_rmse(actuals_orig[:, idx], preds_orig[:, idx])

        # Classification metric for lane change
        actual_lc = np.round(actuals_orig[:, 4]).astype(int)
        pred_lc = np.round(preds_orig[:, 4]).astype(int)
        rmse_all['Lane_Change_Accuracy'] = accuracy_score(actual_lc, pred_lc)

        # Log results
        logging.info(f"Fold {fold} Results: {rmse_all}")
        rmse_results.append({'fold': fold, **rmse_all})
        fold += 1

    # Summary
    logging.info("\nTransformer Metrics across folds:")
    header = f"{'Fold':<6}" + ''.join(f"{t:<20}" for t in standard_targets) + "Haversine_RMSE    LaneChange_Acc"
    logging.info(header)
    for res in rmse_results:
        row = f"{res['fold']:<6}" + ''.join(f"{res.get(t,0):<20.5f}" for t in standard_targets) + \
              f"{res.get('Haversine_RMSE',0):<20.5f}{res.get('Lane_Change_Accuracy',0):<18.4f}"
        logging.info(row)
        
    # mean_v_Vel_rmse = np.mean([r['v_Vel'] for r in rmse_results])
    # all_results.append({**current_params, 'mean_v_Vel_RMSE': mean_v_Vel_rmse})

# # --- Final sorted summary ---
# logging.info("\n📊 Final Results Summary (Sorted by mean v_Vel RMSE):")
# sorted_results = sorted(all_results, key=lambda x: x['mean_v_Vel_RMSE'])
# for res in sorted_results:
#     logging.info(res)
