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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the transformer from CA State Plane III NAD83 (EPSG:2227) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:2227", "EPSG:4326", always_xy=True)


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

def haversine_loss(y_true, y_pred, transformer):
    """
    Compute Haversine loss between actual and predicted coordinates.
    """
    #print("y_true shape:", y_true.shape)  # Debugging
    #print("y_pred shape:", y_pred.shape)  # Debugging

    # if y_true.shape[1] != 4 or y_pred.shape[1] != 4:
    #     raise ValueError(f"Expected y_true and y_pred to have at least 4 columns, but got {y_true.shape}")

    # Transform Global_X, Global_Y to Latitude and Longitude using EPSG transformer
    true_lon, true_lat = transformer.transform(
        y_true[:, 2].cpu().numpy(), y_true[:, 3].cpu().numpy()
    )
    pred_lon, pred_lat = transformer.transform(
        y_pred[:, 2].detach().cpu().numpy(), y_pred[:, 3].detach().cpu().numpy()
    )

    # Clip values for valid range
    true_lat = np.clip(true_lat, -90, 90)
    pred_lat = np.clip(pred_lat, -90, 90)
    true_lon = np.clip(true_lon, -180, 180)
    pred_lon = np.clip(pred_lon, -180, 180)

    # Compute geodesic distances
    distances = np.array([
        geodesic((lat1, lon1), (lat2, lon2)).meters
        for lat1, lon1, lat2, lon2 in zip(true_lat, true_lon, pred_lat, pred_lon)
    ])

    return torch.tensor(np.mean(distances**2), dtype=torch.float32, device=y_true.device)


# Read the first few lines of the CSV file to inspect the structure
# data = pd.read_csv('/kaggle/input/ngsim-vehicle-trajectories/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv', header=0, delimiter=',', )
data1 = pd.read_csv('/DATA/us-dataset/0750_0805_us101_smoothed_21_.csv', header=0, delimiter=',')
data2 = pd.read_csv('/DATA/us-dataset/0805_0820_us101_smoothed_21_.csv', header=0, delimiter=',')
data3 = pd.read_csv('/DATA/us-dataset/0820_0835_us101_smoothed_21_.csv', header=0, delimiter=',')

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
n_lstm_neurons = 32 # Updated hidden size
n_output = 4
num_layers = 2  # Using a 2-layer LSTM
batch_size = 64
epochs = 10
learning_rate = 0.0005
alpha_values = [0.001,0.01]
beta_values = [0.001,0.01]
gamma_values = [0.001,0.01]
delta_values = [0.001,0.01]
sigma_values = [0.001]

# Custom loss printer class for PyTorch
class LossPrinter:
    def __init__(self):
        self.mse_losses_v_Vel = []      # List for MSE losses of v_Vel
        self.mse_losses_Local_Y = []    # List for MSE losses of Local_Y
        self.self_consistency_losses = []  # List for Self-Consistency losses
        #self.mse_losses_Global_X = []   # List for MSE losses of Global_X
        #self.mse_losses_Global_Y = []   # List for MSE losses of Global_Y
        self.haversine_dist_losses = []   # List for Haversine Distance Loss
        self.classification_losses = []    # List for classification loss
        self.congestion_losses = []  # List for congestion loss

    def log_epoch(self, mse_loss_v_Vel, mse_loss_Local_Y,haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss):
        # Store each loss after every epoch
        self.mse_losses_v_Vel.append(mse_loss_v_Vel.item())
        self.mse_losses_Local_Y.append(mse_loss_Local_Y.item())
        #self.mse_losses_Global_X.append(mse_loss_Global_X.item())  # Store Global_X loss
        #self.mse_losses_Global_Y.append(mse_loss_Global_Y.item())  # Store Global_Y loss
        self.haversine_dist_losses.append(haversine_dist_loss.item())  # Store Haversine Distance Loss
        self.self_consistency_losses.append(self_consistency_loss.item())
        self.classification_losses.append(classification_loss.item())
        self.congestion_losses.append(congestion_loss.item())

        # Print the losses for each epoch
        #print(f"  MSE Loss of v_Vel = {mse_loss_v_Vel:.8f}, MSE Loss of Local_Y = {mse_loss_Local_Y:.8f}, Self-Consistency Loss = {self_consistency_loss:.8f}, Classification Loss = {classification_loss:.8f}, Congestion Loss ={congestion_loss:.8f}")
        print(f"MSE Loss of v_Vel = {mse_loss_v_Vel:.8f}, "
              f"MSE Loss of Local_Y = {mse_loss_Local_Y:.8f}, "
              f"Haversine Distance Loss = {haversine_dist_loss:.8f}, "
              f"Self-Consistency Loss = {self_consistency_loss:.8f}, "
              f"Classification Loss = {classification_loss:.8f}, "
              f"Congestion Loss = {congestion_loss:.8f}")
#         # Reset the metrics at the end of each epoch
#         self.mse_metric_v_Vel.reset_state()
#         self.mse_metric_Local_Y.reset_state()
#         self.self_consistency_metric.reset_state()

def custom_loss(y_true, y_pred, class_true, class_pred,congestion_true, congestion_pred, alpha, beta, gamma,delta,sigma,transformer, time_interval=0.1):
    
    # print(f"y_true shape: {y_true.shape}")  # Debugging
    # print(f"y_pred shape: {y_pred.shape}")  # Debugging
    mse_loss_v_Vel = nn.MSELoss()(y_pred[:, 0], y_true[:, 0])
    mse_loss_Local_Y = nn.MSELoss()(y_pred[:, 1], y_true[:, 1])
    #mse_loss_Global_X = nn.MSELoss()(y_pred[:, 2], y_true[:, 2])
    #mse_loss_Global_Y = nn.MSELoss()(y_pred[:, 3], y_true[:, 3])
    haversine_dist_loss = haversine_loss(y_true, y_pred,transformer)
    self_consistency_loss = nn.L1Loss()(y_pred[:, 1], y_true[:, 1] + (y_pred[:, 0] * time_interval))
    classification_loss = nn.CrossEntropyLoss()(class_pred, class_true)
    congestion_loss = nn.MSELoss()(congestion_pred.squeeze(), congestion_true)  # Binary Cross-Entropy for congestion

    #total_loss = 100*mse_loss_v_Vel + (alpha * mse_loss_Local_Y) + (beta * self_consistency_loss) + (gamma * classification_loss) + (delta * congestion_loss) + (sigma * mse_loss_Global_X) + (zeta * mse_loss_Global_Y )
    total_loss = mse_loss_v_Vel + (alpha * mse_loss_Local_Y) + \
                 (beta * self_consistency_loss) + (gamma * classification_loss) + \
                 (delta * congestion_loss) + (sigma * haversine_dist_loss)
    #return total_loss, mse_loss_v_Vel, mse_loss_Local_Y, self_consistency_loss, classification_loss, congestion_loss
    return total_loss, mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss

# # Define your custom loss function with alpha, beta parameters
# def custom_loss(y_true, y_pred, alpha, beta, time_interval=0.1):
# #     print("Shape of y_pred:", y_pred.shape)
# #     print("Shape of y_true:", y_true.shape)
#     mse_loss_v_Vel = nn.MSELoss()(y_pred[:, 0], y_true[:, 0])
#     #print("Calculated MSE Loss for v_Vel:", mse_loss_v_Vel.item())
#     mse_loss_Local_Y = nn.MSELoss()(y_pred[:, 1], y_true[:, 1])

#     # Calculate self-consistency loss
#     actual_Local_Y = y_true[:, 1]
#     predicted_v_Vel = y_pred[:, 0]
#     calculated_Local_Y = actual_Local_Y + (predicted_v_Vel * time_interval)
#     self_consistency_loss = nn.L1Loss()(y_pred[:, 1], calculated_Local_Y)


# #     # Print loss values for debugging purposes
# #     print(f"Calculated MSE Loss for v_Vel: {mse_loss_v_Vel.item():.8f}")
# #     print(f"Calculated MSE Loss for Local_Y: {mse_loss_Local_Y.item():.8f}")
# #     print(f"Calculated Self-Consistency Loss: {self_consistency_loss.item():.8f}")

#     # Combine losses
#     combined_loss = mse_loss_v_Vel + (alpha * mse_loss_Local_Y) + (beta * self_consistency_loss)

#     return combined_loss, mse_loss_v_Vel, mse_loss_Local_Y, self_consistency_loss

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=4, num_layers=2, dropout_rate=0.2, weight_decay=1e-4):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer with dropout (matches LSTM structure)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers (matches LSTM structure)
        self.fc_regression = nn.Linear(hidden_size, output_size)  # Outputs: v_Vel, Local_Y, Global_X, Global_Y
        self.fc_classification = nn.Linear(hidden_size, 3)  # For lane-change classification
        self.fc_congestion = nn.Linear(hidden_size, 1)  # For congestion prediction
        
        # Regularization parameters
        self.weight_decay = weight_decay

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure correct shape

        out, _ = self.gru(x)  # GRU output

        # Apply batch normalization (fixed indexing issue)
        out = self.batch_norm(out[:, -1, :])  # Only take the last timestep

        # Output layers
        regression_output = self.fc_regression(out)
        classification_output = F.log_softmax(self.fc_classification(out), dim=1)
        congestion_output = torch.sigmoid(self.fc_congestion(out))

        return regression_output, classification_output, congestion_output
    
    def get_weight_decay(self):
        return self.weight_decay
# Update metrics
mse_metric_v_Vel.update(mse_loss_v_Vel)
mse_metric_Local_Y.update(mse_loss_Local_Y)
self_consistency_metric.update(self_consistency_loss)
classification_metric.update(classification_loss)

# Metrics for tracking losses
mse_metric_v_Vel = tf.keras.metrics.Mean(name='mse_loss_v_Vel')
mse_metric_Local_Y = tf.keras.metrics.Mean(name='mse_loss_Local_Y')
self_consistency_metric = tf.keras.metrics.Mean(name='self_consistency_loss')
classification_metric =  tf.keras.metrics.Mean(name='classification_loss')



best_val_loss = float('inf')
best_alpha, best_beta, best_gamma, best_delta, best_sigma, best_fold = None, None, None, None, None, None
kf = KFold(n_splits=3)

for alpha, beta, gamma, delta, sigma in product(alpha_values, beta_values, gamma_values, delta_values, sigma_values):
    print(f"Tuning for alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}, sigma={sigma}")

    fold = 1  # Reset fold counter for each new hyperparameter set
    for train_index, test_index in kf.split(features_scaled):
        print(f"Starting Cross-Validation Fold {fold}")

        # Train-validation split
        features_train, features_test = features_scaled[train_index], features_scaled[test_index]
        targets_train, targets_test = target_scaled[train_index], target_scaled[test_index]

        features_train, features_val, targets_train, targets_val = train_test_split(
            features_train, targets_train, test_size=0.2, random_state=42
        )

        # Create PyTorch Dataloaders
        train_dataset = TensorDataset(features_train, targets_train)
        val_dataset = TensorDataset(features_val, targets_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        # Initialize model, loss function, and optimizer
        model = GRUModel(n_features, n_lstm_neurons, n_output).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        loss_printer = LossPrinter()

        # Training Loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

                # Forward pass
                predictions, class_preds, congestion_preds = model(batch_features)

                combined_loss, mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss = custom_loss(
                batch_targets[:, :4], predictions,  
                batch_targets[:, 4].long(), class_preds,  # Lane-Change Labels
                batch_targets[:, 5], congestion_preds,  # Congestion Ground Truth
                alpha, beta, gamma, delta, sigma, transformer
                )

                # Backpropagation
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                train_loss += combined_loss.detach().item()  # Detach for memory efficiency

            # Log loss values
            loss_printer.log_epoch(mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss)

            # Validation loss calculation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    predictions, class_preds, congestion_preds = model(inputs)

                    loss, _, _, _, _, _, _ = custom_loss(
                    targets[:, :4], predictions,
                    targets[:, 4].long(), class_preds,
                    targets[:, 5], congestion_preds,
                    alpha, beta, gamma, delta,sigma,transformer
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)  # Normalize validation loss

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # Check for best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_alpha, best_beta, best_gamma, best_delta, best_sigma = alpha, beta, gamma, delta, sigma
                best_fold = fold
                print(f"New best loss: {best_val_loss} with alpha={best_alpha}, beta={best_beta}, gamma={best_gamma}, delta={best_delta}, sigma={best_sigma} on fold {best_fold}")

        fold += 1

print(f"Best alpha: {best_alpha}, Best beta: {best_beta}, Best Gamma: {best_gamma}, Best Delta: {best_delta}, Best Sigma: {best_sigma}, Best Fold: {best_fold}, Best Validation Loss: {best_val_loss}")
# Ensure directory for saving weights
checkpoint_dir = "/kaggle/working/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon):
    distances = [
        geodesic((lat1, lon1), (lat2, lon2)).meters
        for lat1, lon1, lat2, lon2 in zip(actual_lat, actual_lon, pred_lat, pred_lon)
    ]
    return np.sqrt(np.mean(np.square(distances)))  # RMSE of distances

# Example
best_alpha = 0.01
best_beta = 0.01
best_gamma = 0.01
best_delta = 0.01
best_sigma = 0.01

# Cross-validation setup
kf = KFold(n_splits=3, shuffle=False)
rmse_results = []
loss_printer_callback = {
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
fold = 1

for train_index, test_index in kf.split(features_scaled):
    print(f"Starting Cross-Validation Fold {fold}")

    # Split into training and testing sets
    features_train, features_test = features_scaled[train_index], features_scaled[test_index]
    targets_train, targets_test = target_scaled[train_index], target_scaled[test_index]

    # Further split training set into training and validation
    features_train, features_val, targets_train, targets_val = train_test_split(
        features_train, targets_train, test_size=0.2, random_state=42
    )

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(features_train), torch.FloatTensor(targets_train))
    val_dataset = TensorDataset(torch.FloatTensor(features_val), torch.FloatTensor(targets_val))
    test_dataset = TensorDataset(torch.FloatTensor(features_test), torch.FloatTensor(targets_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model and optimizer
    model = GRUModel(input_size=features_train.shape[1], hidden_size=n_lstm_neurons, output_size=4,num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=model.get_weight_decay())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Initialize fold-wise storage at the start of each fold
    loss_printer_callback[fold] = {
        'total_losses': [],
        'mse_losses_v_Vel': [],
        'mse_losses_Local_Y': [],
        'haversine_dist_losses': [],
        'self_consistency_losses': [],
        'classification_losses': [],
        'congestion_losses': []
    }

    # Train the model
    for epoch in tqdm(range(10)):
        model.train()
        epoch_loss = 0.0
        sum_mse_v_Vel = 0.0
        sum_mse_local_Y = 0.0
        sum_haversine = 0.0
        sum_self_consistency = 0.0
        sum_classification = 0.0
        sum_congestion = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions, class_preds, congestion_preds = model(inputs)

            # Compute Loss
            loss, mse_loss_v_Vel, mse_loss_Local_Y, haversine_dist_loss, self_consistency_loss, classification_loss, congestion_loss = custom_loss(
            labels[:, :4], predictions,  
            labels[:, 4].long(), class_preds,  # Lane-Change Labels
            labels[:, 5], congestion_preds,  # Congestion Labels
            best_alpha, best_beta, best_gamma, best_delta, best_sigma, transformer
        )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()  # Detach for memory efficiency
            sum_mse_v_Vel += mse_loss_v_Vel.item()
            sum_mse_local_Y += mse_loss_Local_Y.item()
            sum_haversine += haversine_dist_loss.item()
            sum_self_consistency += self_consistency_loss.item()
            sum_classification += classification_loss.item()
            sum_congestion += congestion_loss.item()
            
        avg_mse_v = sum_mse_v_Vel / len(train_loader)
        avg_mse_y = sum_mse_local_Y / len(train_loader)
        avg_haversine = sum_haversine / len(train_loader)
        avg_self_consistency = sum_self_consistency / len(train_loader)
        avg_classification = sum_classification / len(train_loader)
        avg_congestion = sum_congestion / len(train_loader)

        # Log loss values for analysis
        loss_printer_callback[fold]['total_losses'].append(epoch_loss / len(train_loader))
        loss_printer_callback[fold]['mse_losses_v_Vel'].append(avg_mse_v)
        loss_printer_callback[fold]['mse_losses_Local_Y'].append(avg_mse_y)
        loss_printer_callback[fold]['haversine_dist_losses'].append(avg_haversine)
        loss_printer_callback[fold]['self_consistency_losses'].append(avg_self_consistency)
        loss_printer_callback[fold]['classification_losses'].append(avg_classification)
        loss_printer_callback[fold]['congestion_losses'].append(avg_congestion)


        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.8f},"
              f"MSE Loss of v_Vel = {mse_loss_v_Vel:.8f},"
              f"MSE Loss of Local_Y = {mse_loss_Local_Y:.8f},"
              f"Haversine Distance Loss = {haversine_dist_loss:.8f},"
              f"Self-Consistency Loss = {self_consistency_loss:.8f}, "
              f"Classification Loss = {classification_loss:.8f},"
              f"Congestion Loss = {congestion_loss:.8f}")
        
        # Adjust learning rate
        scheduler.step(epoch_loss / len(train_loader))

        # Save model weights
        torch.save(model.state_dict(), f'model_fold{fold}_epoch{epoch + 1}.pth')

    # Model Evaluation on Test Data
    model.eval()
    predictions = []
    lane_change_preds = []
    congestion_preds = []
    actual_v_Vel_list = []
    actual_local_Y_list = []
    actual_Global_X_list =[]
    actual_Global_Y_list = []
    actual_lane_change_list = []
    actual_congestion_list = []
    actual_lat_list = []
    actual_lon_list = []
    pred_lat_list = []
    pred_lon_list = []
    vehicle_ids_list = []
    simulation_times_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward Pass
            predictions_batch, class_preds_batch, congestion_preds_batch = model(inputs)

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


            # Store regression outputs
            predictions.append(predictions_batch.cpu())   # Keep on GPU
            actual_v_Vel_list.append(labels[:, 0].cpu().numpy())
            actual_local_Y_list.append(labels[:, 1].cpu().numpy())

            # actual_Global_X_list.append(labels[:, 2].cpu().numpy())
            # actual_Global_Y_list.append(labels[:, 3].cpu().numpy()) 
            # # Store classification outputs
            lane_change_preds.append(class_preds_batch.cpu().numpy())  # Move to CPU
            actual_lane_change_list.append(labels[:, 4].cpu().numpy())  # Lane change labels

            # Store congestion outputs
            congestion_preds.append(congestion_preds_batch.cpu().numpy())
            actual_congestion_list.append(labels[:, 5].cpu().numpy())

    # Concatenate predictions
    predictions = torch.cat(predictions).cpu().numpy()  # Move predictions to CPU for further evaluation
    lane_change_preds = np.concatenate(lane_change_preds)  # Convert to numpy array
    actual_v_Vel = np.concatenate(actual_v_Vel_list)
    actual_local_Y = np.concatenate(actual_local_Y_list)
    actual_lat = np.array(actual_lat_list)  # Convert latitudes to numpy array
    actual_lon = np.array(actual_lon_list)  # Convert longitudes to numpy array
    pred_lat = np.array(pred_lat_list)  # Convert predicted latitudes to numpy array
    pred_lon = np.array(pred_lon_list)  # Convert predicted longitudes to numpy array
    actual_lane_change = np.concatenate(actual_lane_change_list)
    congestion_preds = np.concatenate(congestion_preds)
    actual_congestion = np.concatenate(actual_congestion_list)

    # Convert lane-change probabilities to class labels
    predicted_lane_change = np.argmax(lane_change_preds, axis=1)  # Get class with highest probability
    actual_lane_change = actual_lane_change.astype(int)  # Convert actual values to integers
    predicted_lane_change = predicted_lane_change.astype(int)  # Convert predicted values to integers

    # Calculate RMSE for velocity prediction
    rmse_v_Vel = calculate_rmse(actual_v_Vel, predictions[:, 0])
    print(f"RMSE for velocity prediction: {rmse_v_Vel}")

    # Calculate RMSE for local Y prediction
    rmse_local_Y = calculate_rmse(actual_local_Y, predictions[:, 1])
    print(f"RMSE for local Y prediction: {rmse_local_Y}")

    # Compute RMSE for latitude and longitude
    rmse_haversine = haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon)
    print(f"Haversine RMSE (meters): {rmse_haversine}")

    
    # Compute RMSE for congestion prediction
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


    # Store the RMSE values for this fold
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
        'predicted_lane': predicted_lane_change  # Store predicted lane
    })

    fold += 1

# Print a summary table of RMSE for all folds
print("\nSummary of RMSE GRU across folds:")
print(f"{'Fold':<5} {'RMSE for velocity':<20} {'RMSE for Local_Y':<20} {'Haversine RMSE':<20} {'RMSE for congestion':<20}   {'lane_change_accuracy':<20}  {'pred_lat':<20}  {'pred_lon':<20}  {'predicted_lane':<15}")
for result in rmse_results:
    avg_pred_lat = np.mean(result['pred_lat'])  # Take mean predicted latitude
    avg_pred_lon = np.mean(result['pred_lon'])  # Take mean predicted longitude
    avg_pred_lane = np.mean(result['predicted_lane'])  # Average predicted lane (for summary)
    print(f"{result['fold']:<5} {result['rmse_v_Vel']:<20.5f} {result['rmse_local_Y']:<20.5f} {result['Haversine RMSE']:<20.5f} {result['rmse_congestion']:<20.5f} {result['lane_change_accuracy']:<20.5f}  {avg_pred_lat:<15.5f} {avg_pred_lon:<15.5f} {avg_pred_lane:<15.2f}")

# # Adjust global font sizes for readability
# plt.rcParams.update({
#     'axes.titlesize': 18,      # Font size for plot title
#     'axes.labelsize': 18,      # Font size for x and y axis labels
#     'xtick.labelsize': 14,     # Font size for x axis tick labels
#     'ytick.labelsize': 14,     # Font size for y axis tick labels
#     'legend.fontsize': 14,     # Font size for legend labels
#     'figure.figsize': (14, 10) # Adjust figure size (optional)
# })

# # Plotting the total loss
# plt.plot(loss_printer_callback['total_losses'])
# plt.title('Total Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.savefig('/home/m23air002/total_loss_plot(GRU).png')
# plt.show()

# plt.figure()

# # Plot MSE Loss
# plt.plot(loss_printer_callback['mse_losses_v_Vel'], label='MSE Loss v_Vel')
# plt.plot(loss_printer_callback['mse_losses_Local_Y'], label='MSE Loss Local_Y')
# plt.plot(loss_printer_callback['haversine_dist_losses'],label='haversine_dist_losses')
# plt.plot(loss_printer_callback['congestion_losses'], label='congestion_losses')
# plt.plot(loss_printer_callback['classification_losses'], label='classification_losses')


# # Plot Self-Consistency Loss
# plt.plot(loss_printer_callback['self_consistency_losses'], label='Self-Consistency Loss')

# # Add labels and title
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('MSE and Self-Consistency Losses over Epochs')
# plt.legend()
# plt.savefig('/home/m23air002/MSE_and_Self-Consistency_Losses(GRU).png')
# plt.show()

# #Ensure your predicted and actual values are NumPy arrays for consistent plotting
# actual_v_Vel = np.array(actual_v_Vel)
# predicted_v_Vel = np.array(predictions[:, 0])
# actual_local_Y = np.array(actual_local_Y)
# predicted_local_Y = np.array(predictions[:, 1])
# actual_lat = np.array(actual_lat)
# predicted_lat = np.array(predictions[:, 2])  # Assuming index 2 is latitude
# actual_lon = np.array(actual_lon)
# predicted_lon = np.array(predictions[:, 3])  # Assuming index 3 is longitude



# # Set the figure size for better readability
# plt.figure(figsize=(14, 12))

# # Plot for v_Vel
# #plt.subplot(3, 2, 1)
# plt.plot(actual_v_Vel, label='Actual v_Vel', color='blue', alpha=0.7, linestyle='-')
# plt.plot(predicted_v_Vel, label='Predicted v_Vel', color='orange', alpha=0.7, linestyle='--')
# plt.title('Predicted vs Actual v_Vel')
# plt.xlabel('Samples')
# plt.ylabel('v_Vel')
# plt.legend()

# # Plot for Local_Y
# #plt.subplot(3, 2, 2)
# plt.figure(figsize=(14, 12))
# plt.plot(actual_local_Y, label='Actual Local_Y', color='green', alpha=0.7, linestyle='-')
# plt.plot(predicted_local_Y, label='Predicted Local_Y', color='red', alpha=0.7, linestyle='--')
# plt.title('Predicted vs Actual Local_Y')
# plt.xlabel('Samples')
# plt.ylabel('Local_Y')
# plt.legend()

# # Plot for congestion
# #plt.subplot(3, 2, 3)
# plt.figure(figsize=(14, 12))
# plt.plot(actual_congestion, label='Actual Congestion', color='purple', alpha=0.7, linestyle='-')
# plt.plot(congestion_preds, label='Predicted Congestion', color='brown', alpha=0.7, linestyle='--')
# plt.title('Predicted vs Actual Congestion')
# plt.xlabel('Samples')
# plt.ylabel('Congestion')
# plt.legend()

# # Plot for Latitude
# #plt.subplot(3, 2, 4)
# plt.figure(figsize=(14, 12))
# plt.plot(actual_latitude, label='Actual Latitude', color='cyan', alpha=0.7, linestyle='-')
# plt.plot(predicted_latitude, label='Predicted Latitude', color='magenta', alpha=0.7, linestyle='--')
# plt.title('Predicted vs Actual Latitude')
# plt.xlabel('Samples')
# plt.ylabel('Latitude')
# plt.legend()

# # Plot for Longitude
# #plt.subplot(3, 2, 5)
# plt.figure(figsize=(14, 12))
# plt.plot(actual_longitude, label='Actual Longitude', color='black', alpha=0.7, linestyle='-')
# plt.plot(predicted_longitude, label='Predicted Longitude', color='yellow', alpha=0.7, linestyle='--')
# plt.title('Predicted vs Actual Longitude')
# plt.xlabel('Samples')
# plt.ylabel('Longitude')
# plt.legend()

# # Adjust layout for better visibility
# plt.tight_layout()

# # Save the plots
# plt.savefig('/home/m23air002/predictions_vs_actuals(GRU).png')

# # Show the plots
# plt.show()

# # Assuming rmse_results is a list of tuples or lists like [(fold, rmse_v_Vel, rmse_local_Y), ...]
# rmse_results_df = pd.DataFrame(rmse_results, columns=['Fold', 'RMSE for velocity', 'RMSE for Local_Y','RMSE for Congestion', 'lane_change_accuracy','lane_change_precision', 'lane_change_recall', 'lane_change_f1','actual_lat','actual_lon','pred_lat','pred_lon','predicted_lane' ])

# # Save the DataFrame to a CSV file
# rmse_results_df.to_csv('GRU_results.csv', index=False)

