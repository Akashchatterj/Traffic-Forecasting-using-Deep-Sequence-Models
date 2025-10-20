import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
#from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#from tensorflow.keras.models import Sequential,load_model
#from tensorflow.keras.regularizers import l2
import warnings
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from pyproj import Transformer
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
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transformer = Transformer.from_crs("epsg:32611", "epsg:4326", always_xy=True)

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

    # Resample to 1-second intervals explicitly
    data = data.resample('1S', on='Global_Time').mean().dropna().reset_index()

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
data1 = pd.read_csv('/DATA2/usdataset/0750_0805_us101_smoothed_21_.csv', header=0, delimiter=',')
data2 = pd.read_csv('/DATA2/usdataset/0805_0820_us101_smoothed_21_.csv', header=0, delimiter=',')
data3 = pd.read_csv('/DATA2/usdataset/0820_0835_us101_smoothed_21_.csv', header=0, delimiter=',')

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
final_feature_columns = ['Local_X', 'Lane_ID', 'v_Vel', 'v_Acc', 'v_Length', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6']
target_columns = ['v_Vel', 'Local_Y', 'Global_X', 'Global_Y', 'Lane_Change_Label', 'JF']

# Prepare Data
scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_target = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler_features.fit_transform(data_combined[final_feature_columns])
target_scaled = scaler_target.fit_transform(data_combined[target_columns])

features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
target_scaled = torch.tensor(target_scaled, dtype=torch.float32)

# features_train = scaler_features.fit_transform(train_data[final_feature_columns])
# target_train = scaler_target.fit_transform(train_data[target_columns])

# features_test = scaler_features.transform(test_data[final_feature_columns])
# target_test = scaler_target.transform(test_data[target_columns])

# Convert to tensors
# features_train = torch.tensor(features_train, dtype=torch.float32)
# target_train = torch.tensor(target_train, dtype=torch.float32)
# features_test = torch.tensor(features_test, dtype=torch.float32)
# target_test = torch.tensor(target_test, dtype=torch.float32)


# Hyperparameters
n_input = 10
n_features = len(final_feature_columns)
n_output = len(target_columns)
batch_size = 16
epochs = 50
learning_rate = 0.001
n_lstm_neurons = 10

# Custom loss printer class for PyTorch
class LossPrinter:
    def __init__(self):
        self.losses = []      # List for MSE losses of v_Vel
        #self.mse_losses_Local_Y = []    # List for MSE losses of Local_Y
        #self.self_consistency_losses = []  # List for Self-Consistency losses

    def log_epoch(self, mse_loss_v_Vel, mse_loss_Local_Y, self_consistency_loss):
        # Store each loss after every epoch
        self.losses.append(total_loss.item())
        #self.mse_losses_Local_Y.append(mse_loss_Local_Y.item())
        #self.self_consistency_losses.append(self_consistency_loss.item())

        # Print the losses for each epoch
        print(f"  Total MSE Loss = {total_loss:.8f}")

def custom_loss(y_true, y_pred):
    mse_loss = nn.MSELoss()
    loss = mse_loss(y_pred, y_true)
    return loss

# PyTorch model (an GRU)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure x has shape (batch_size, sequence_length, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence length dimension if missing

        out, _ = self.lstm(x)  # LSTM output should now be (batch_size, sequence_length, hidden_dim)
        out = self.fc(out[:, -1, :])  # Take the last time step if sequence length > 1
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
# X_train, y_train = create_sequences(features_train, target_train, n_input)
# X_test, y_test = create_sequences(features_test, target_test, n_input)

# Grid of hyperparameters
param_grid = {
   'batch_size': [6, 8, 10],
   'learning_rate': [5e-3, 1e-4, 5e-4],
   'n_lstm_neurons': [10, 12, 14],
   'n_input': [5, 8, 10],
   'epochs': [10],  
}

hyperparam_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

# Store full summary
all_results = []

loss_printer_callback = {}

for combo in hyperparam_combinations:
    torch.manual_seed(133)
    #print(f"\n🔧 Training with hyperparameters: {current_params}")
    
    current_params = dict(zip(param_keys, combo))

    # Extract current hyperparameters
    batch_size = current_params['batch_size']
    learning_rate = current_params['learning_rate']
    n_lstm_neurons = current_params['n_lstm_neurons']
    n_input = current_params['n_input']
    epochs = current_params['epochs']

    print(f"\n\U0001F50D Testing config: Past Window Size={n_input}, Learning Rate={learning_rate}, Batch Size={batch_size}, Num of Epochs={epochs}")
    
    # Re-create sequence inputs with current n_input
    X, y = create_sequences(features_scaled, target_scaled, n_input)
    
    kf = KFold(n_splits=3, shuffle=False)
    rmse_results = []
    
    fold = 1

    # --- Define Target Groups ---
    geo_targets = ['Global_X', 'Global_Y']
    standard_targets = ['v_Vel', 'Local_Y', 'JF']

    for train_index, test_index in kf.split(X):
        print(f"\nStarting Cross-Validation Fold {fold}")

        # Initialize loss printer callback for current fold
        loss_printer_callback[fold] = {
            'epoch_losses': [],
        }

        # Split data into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Further split training data into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Prepare DataLoader objects
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        # Initialize model
        model = GRUModel(input_size=n_features, hidden_size=n_lstm_neurons, output_size=len(target_columns)).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in tqdm(range(epochs), desc=f"Fold {fold}"):
            model.train()
            epoch_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.6f}")

            # Store loss for the current epoch in current fold
            loss_printer_callback[fold]['epoch_losses'].append(avg_epoch_loss)

            # Save model weights
            #torch.save(model.state_dict(), f'model_fold{fold}_epoch{epoch + 1}.pth')

        # Evaluation on test data
        model.eval()
        predictions = []
        actuals = []
        actual_lat_list = []
        actual_lon_list = []
        pred_lat_list = []
        pred_lon_list = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # # Convert actual Global_X, Global_Y to Lat/Lon
                # for i in range(len(labels)):
                #     global_x, global_y = labels[i, 2].cpu().item(), labels[i, 3].cpu().item()
                #     lon, lat = transformer.transform(global_x, global_y)
                #     actual_lat_list.append(lat)
                #     actual_lon_list.append(lon)
                # # Convert predicted Global_X, Global_Y to Lat/Lon
                # for i in range(len(predictions_batch)):
                #     pred_x, pred_y = predictions_batch[i, 2].cpu().item(), predictions_batch[i, 3].cpu().item()
                #     pred_lon, pred_lat = transformer.transform(pred_x, pred_y)
                #     pred_lat_list.append(pred_lat)
                #     pred_lon_list.append(pred_lon)

                predictions.append(outputs.cpu().numpy())
                actuals.append(labels.cpu().numpy())
            

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        # actual_lat = np.array(actual_lat_list)  # Convert latitudes to numpy array
        # actual_lon = np.array(actual_lon_list)  # Convert longitudes to numpy array
        # pred_lat = np.array(pred_lat_list)  # Convert predicted latitudes to numpy array
        # pred_lon = np.array(pred_lon_list)  # Convert predicted longitudes to numpy array

        # Inverse transform the predicted and actual values
        predicted_all_orig = scaler_target.inverse_transform(predictions)
        actual_all_orig = scaler_target.inverse_transform(actuals)

        # --- Convert Global_X and Global_Y (columns 2 and 3) into Lat/Lon ---
        # For actual values
        actual_global_x = actual_all_orig[:, 2]
        actual_global_y = actual_all_orig[:, 3]
        actual_lon, actual_lat = transformer.transform(actual_global_x, actual_global_y)

        # For predicted values
        pred_global_x = predicted_all_orig[:, 2]
        pred_global_y = predicted_all_orig[:, 3]
        pred_lon, pred_lat = transformer.transform(pred_global_x, pred_global_y)


        # Initialize dictionary to store RMSE values per target
        rmse_all_targets = {}
        haversine_computed = False

        # Calculate RMSE for each target column
        for idx, col in enumerate(target_columns):
            if col in geo_targets:
                if not haversine_computed:
                    rmse_haversine = haversine_rmse(actual_lat, actual_lon, pred_lat, pred_lon)
                    rmse_all_targets['Haversine_RMSE'] = rmse_haversine
                    print(f"Fold {fold} Haversine RMSE (Global_X/Y): {rmse_haversine:.5f}")
                    haversine_computed = True
            else:
                rmse_value = calculate_rmse(actual_all_orig[:, idx], predicted_all_orig[:, idx])
                rmse_all_targets[col] = rmse_value
                print(f"Fold {fold} RMSE for {col}: {rmse_value:.5f}")
                
        # --- Lane Change Prediction Metrics ---
        actual_lane_change = np.round(actual_all_orig[:, 4]).astype(int)
        predicted_lane_change = np.round(predicted_all_orig[:, 4]).astype(int)
        accuracy = accuracy_score(actual_lane_change, predicted_lane_change)
        print(f"Fold {fold} Lane Change Accuracy: {accuracy:.4f}")
        # Add classification metrics to results   
        rmse_all_targets['Lane_Change_Accuracy'] = accuracy
        #rmse_results.append({'fold': fold, **rmse_all_targets})
            
            

        # Then compute RMSE in original scale
        #rmse_v_Vel_original = calculate_rmse(actual_v_Vel_orig, predicted_v_Vel_orig)

        #rmse_v_Vel = calculate_rmse(actuals, predictions)
        #print(f"Fold {fold} RMSE for velocity prediction: {rmse_v_Vel:.5f}")

        rmse_results.append({
            'fold': fold,
            **rmse_all_targets
        })

        fold += 1

    # RMSE Summary for all targets
    print("\nSummary of Metrics Simple GRU across folds:")
    header = f"{'Fold':<6} " + " ".join([f"{col:<20}" for col in standard_targets]) + "Haversine_RMSE     LaneChange_Acc"
    print(header)

    for result in rmse_results:
        row = f"{result['fold']:<6} " + " ".join([f"{result.get(col, 0):<20.5f}" for col in standard_targets]) + f"{result.get('Haversine_RMSE', 0):<20.5f}" + f"{result.get('Lane_Change_Accuracy', 0):<18.4f}"
        print(row)

    mean_v_Vel_rmse = np.mean([r['v_Vel'] for r in rmse_results])
    all_results.append({**current_params, 'mean_v_Vel_RMSE': mean_v_Vel_rmse})

# Final sorted summary
print("\n\U0001F4CA Final Results Summary (Sorted by mean v_Vel RMSE):")
sorted_results = sorted(all_results, key=lambda x: x['mean_v_Vel_RMSE'])
for res in sorted_results:
    print(res)
# # Optional: Visualize fold-wise epoch losses
# for fold_num, losses in loss_printer_callback.items():
#     plt.plot(losses['epoch_losses'], label=f'Fold {fold_num}')

# plt.xlabel('Epochs')
# plt.ylabel('Average Training Loss')
# plt.title('Training Loss per Fold')
# plt.legend()
# plt.savefig("/home/m23air002/simple_lstm_loss")
# plt.show()
# plt.figure()

# # Ensure NumPy arrays for plotting
# actual_v_Vel = actuals[:, 0]
# predicted_v_Vel = predictions[:, 0]
# #actual_v_Vel = np.array(actuals).flatten()
# #predicted_v_Vel = np.array(predictions).flatten()

# # Set figure size
# plt.figure(figsize=(12, 6))

# # Plot predicted vs actual velocity (v_Vel)
# plt.plot(actual_v_Vel, label='Actual v_Vel', color='blue', linestyle='-', alpha=0.8)
# plt.plot(predicted_v_Vel, label='Predicted v_Vel', color='orange', linestyle='--', alpha=0.8)

# # Titles and labels
# plt.title('Predicted vs Actual Vehicle Velocity (v_Vel)', fontsize=14)
# plt.xlabel('Sample Index', fontsize=12)
# plt.ylabel('Velocity (scaled)', fontsize=12)
# plt.legend()
# plt.grid(True, alpha=0.3)

# # Save figure clearly
# plt.savefig('/home/m23air002/predictions_vs_actuals_velocity_simple_lstm.png')

# # Show plot
# plt.tight_layout()
# plt.show()

# Correctly structured DataFrame for velocity-only predictions
#rmse_results_df = pd.DataFrame(rmse_results, columns=['Fold', 'RMSE for velocity'])

# Save to CSV with clear naming
#rmse_results_df.to_csv('rmse_results_velocity_only.csv', index=False)

