from torch.utils.data import Dataset
import torch
import numpy as np
import json
import os
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import random

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
FREQUENCY = 60  # 60 seconds

def create_datasets(input_files='data/processed', ship_type='Cargo', with_full_data=False, json_dir=None):
    # 1. Discover parquet files
    parquet_files = _discover_parquet_files(input_files, ship_type)

    # 2. load data and process into sequences
    all_trajectories = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df = df.dropna().reset_index(drop=True)
        df = df[['Timestamp', 'Latitude', 'Longitude', 'SOG', 'COG']]
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        all_trajectories.append(df)

    print(f"Total loaded trajectories: {len(all_trajectories)}")
    
    # 3. Resample trajectories to regulize time series
    resampled_trajectories = []
    for traj in all_trajectories:
        resampled = _resample_trajectory(traj)
        if resampled is None:
            continue
        if isinstance(resampled, list):
            resampled_trajectories.extend(resampled)
        else:
            resampled_trajectories.append(resampled)


    # # create 4 suplots in a 2x2 grid for a sample trajectory
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # # pick out one trajectory for debugging
    # my_trajectory_index = random.randint(0, len(resampled_trajectories)-1)
    # # plot the trajectory in first subplot
    # axs[0, 0].plot(resampled_trajectories[my_trajectory_index]['Longitude'], resampled_trajectories[my_trajectory_index]['Latitude'], marker='o')
    # axs[0, 0].set_title('Sample Resampled Trajectory')
    # axs[0, 0].set_xlabel('Longitude')
    # axs[0, 0].set_ylabel('Latitude')
    # axs[0, 0].grid()


    print(f"Total resampled trajectories: {len(resampled_trajectories)}")
    
    # 4. Split COG into sine and cosine components
    for i in range(len(resampled_trajectories)):
        df = resampled_trajectories[i]
        cog_rad = np.deg2rad(df['COG'].values)
        df['COG_sin'] = np.sin(cog_rad)
        df['COG_cos'] = np.cos(cog_rad)
        df = df.drop(columns=['COG'])
        resampled_trajectories[i] = df

    # 5. Determine reference lat/lon for conversion or load existing
    # filepath
    filepath = os.path.join(json_dir, "reference_latlon.json")
    if with_full_data:
        all_lats = np.concatenate([traj['Latitude'].values for traj in resampled_trajectories])
        mean_lat = np.mean(all_lats)
        all_lons = np.concatenate([traj['Longitude'].values for traj in resampled_trajectories])
        mean_lon = np.mean(all_lons)
        # store mean lat/lon for future use
        with open(filepath, "w") as f:
            json.dump({
                "mean_latitude": mean_lat,
                "mean_longitude": mean_lon
            }, f)
    elif os.path.exists(filepath):
        with open(filepath, "r") as f:
            ref = json.load(f)
        mean_lat = ref["mean_latitude"]
        mean_lon = ref["mean_longitude"]
    else:
        raise ValueError("Reference latitude and longitude file not found. Set with_full_data=True to create it.")

    # 6. Convert Lat and Lon to X and Y
    for i in range(len(resampled_trajectories)):
        resampled_trajectories[i] = _convert_latlon_to_xy(resampled_trajectories[i], mean_lat, mean_lon)

    # # Debug: plot the same trajectory after conversion in the second subplot
    # axs[0, 1].plot(resampled_trajectories[my_trajectory_index]['X'], resampled_trajectories[my_trajectory_index]['Y'], marker='o', color='orange')
    # axs[0, 1].set_title('Sample Trajectory after Lat/Lon to X/Y Conversion')
    # axs[0, 1].set_xlabel('X (meters)')
    # axs[0, 1].set_ylabel('Y (meters)')
    # axs[0, 1].grid()

    # # Debug: plot the same tajectory converted back to lat/lon in the third subplot
    # sample_xy = resampled_trajectories[my_trajectory_index][['X', 'Y']].values
    # converted_back = to_latlon(sample_xy)
    # axs[1, 0].plot(converted_back[:,0], converted_back[:,1], marker='o', color='green')
    # axs[1, 0].set_title('Sample Trajectory after Converting Back to Lat/Lon')
    # axs[1, 0].set_xlabel('Longitude')
    # axs[1, 0].set_ylabel('Latitude')
    # axs[1, 0].grid()
    # plt.show()

    # 7. Add delta X and delta Y columns + remove NaNs
    for i in range(len(resampled_trajectories)):
        df = resampled_trajectories[i]
        df['Delta_X'] = df['X'].diff()
        df['Delta_Y'] = df['Y'].diff()
        df = df.iloc[1:]   # drop the first row (index 0)
        resampled_trajectories[i] = df
    
    # 8. Assign sequences to train, val, test sets
    train_trajectories, temp_trajectories = train_test_split(resampled_trajectories, train_size=TRAIN_RATIO, random_state=42)
    val_trajectories, test_trajectories = train_test_split(temp_trajectories, test_size=TEST_RATIO/(TEST_RATIO + VAL_RATIO), random_state=42)

    # 9. Compute and store normalization statistics from training data if working on full data
    if with_full_data:
        _compute_and_store_norm_stats(train_trajectories, json_dir)

    # # create 4 suplots in a 2x2 grid for a sample trajectory
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # # plot a sample trajectory before normalization in the first subplot
    # random_index = random.randint(0, len(train_trajectories)-1)
    # sample_traj = train_trajectories[random_index]
    # axs[0, 0].plot(sample_traj['X'], sample_traj['Y'], marker='o')
    # axs[0, 0].set_title('Sample Training Trajectory before Normalization (X vs Y)')
    # axs[0, 0].set_xlabel('X (meters)')
    # axs[0, 0].set_ylabel('Y (meters)')
    # axs[0, 0].grid()

    # 10. Apply normalization to all datasets
    train_trajectories = _apply_normalization(train_trajectories, json_dir)
    val_trajectories = _apply_normalization(val_trajectories, json_dir)
    test_trajectories = _apply_normalization(test_trajectories, json_dir)

    # # Debug: plot the same trajectory after normalization
    # sample_traj_norm = train_trajectories[random_index]
    # axs[0, 1].plot(sample_traj_norm['X'], sample_traj_norm['Y'], marker='o', color='orange')
    # axs[0, 1].set_title('Sample Training Trajectory after Normalization (X vs Y)')
    # axs[0, 1].set_xlabel('X (normalized)')
    # axs[0, 1].set_ylabel('Y (normalized)')
    # axs[0, 1].grid()

    # # plot the same trajectory converted back to denormalized X,Y
    # denorm_xy = denormalize_xy(sample_traj_norm[['X', 'Y']].values)
    # axs[1, 0].plot(denorm_xy[:,0], denorm_xy[:,1], marker='o', color='green')
    # axs[1, 0].set_title('Sample Training Trajectory after Denormalization (X vs Y)')
    # axs[1, 0].set_xlabel('X (meters)')
    # axs[1, 0].set_ylabel('Y (meters)')
    # axs[1, 0].grid()
    # plt.show()

    # 10.5 reorder columns to have X,Y,Delta_X,Delta_Y,SOG,COG_sin,COG_cos
    def _reorder_columns(df):
        return df[['X', 'Y', 'Delta_X', 'Delta_Y', 'SOG', 'COG_sin', 'COG_cos']]

    train_trajectories = [_reorder_columns(df) for df in train_trajectories]
    val_trajectories = [_reorder_columns(df) for df in val_trajectories]
    test_trajectories = [_reorder_columns(df) for df in test_trajectories]

    # 11. Create subsequences of length 30
    train_sequences = _create_subsequences(train_trajectories, seq_length=30)
    val_sequences = _create_subsequences(val_trajectories, seq_length=30)
    test_sequences = _create_subsequences(test_trajectories, seq_length=30)

    # 12. Create Dataset objects
    train_data = AISDataset(train_sequences)
    val_data = AISDataset(val_sequences)
    test_data = AISDataset(test_sequences)

    return train_data, val_data, test_data

def to_latlon(xy, json_dir):
    """
    Convert X,Y coordinates in meters back to Latitude,Longitude.
    Accepts:
        xy: shape (2,) or (N,2)
    Returns:
        latlon: shape (1,2) or (N,2)
    """
    xy = np.asarray(xy)

    # Ensure shape (N,2)
    if xy.ndim == 1:
        if xy.shape[0] != 2:
            raise ValueError("Input must be shape (2,) or (N,2)")
        xy = xy.reshape(1, 2)
    elif xy.ndim == 2:
        if xy.shape[1] != 2:
            raise ValueError("Input must be shape (2,) or (N,2)")
    else:
        raise ValueError("Input must be 1D or 2D NumPy array")

    x = xy[:, 0]
    y = xy[:, 1]

    latlon_filepath = os.path.join(json_dir, "reference_latlon.json")

    with open(latlon_filepath, "r") as f:
        ref = json.load(f)
    ref_lat = ref["mean_latitude"]
    ref_lon = ref["mean_longitude"]

    R = 6371000
    lat_rad = y / R + np.deg2rad(ref_lat)
    lon_rad = x / (R * np.cos(np.deg2rad(ref_lat))) + np.deg2rad(ref_lon)

    lat = np.rad2deg(lat_rad)
    lon = np.rad2deg(lon_rad)

    return np.column_stack((lon, lat))

def denormalize_xy(xy, json_dir):
    """
    Denormalize stacked X,Y coordinates.
    Accepts:
        xy: shape (2,) or (N,2)
    Returns:
        xy_denorm: shape (1,2) or (N,2)
    """

    xy = np.asarray(xy)

    # Ensure shape (N,2)
    if xy.ndim == 1:
        if xy.shape[0] != 2:
            raise ValueError("Input must be shape (2,) or (N,2)")
        xy = xy.reshape(1, 2)
    elif xy.ndim == 2:
        if xy.shape[1] != 2:
            raise ValueError("Input must be shape (2,) or (N,2)")
    else:
        raise ValueError("Input must be 1D or 2D NumPy array")

    x = xy[:, 0]
    y = xy[:, 1]
    
    filepath = os.path.join(json_dir, "train_data_statistics.json")
    train_means, train_stds = load_norm_stats(filepath)

    x_denorm = x * train_stds[0] + train_means[0]
    y_denorm = y * train_stds[1] + train_means[1]

    return np.column_stack((x_denorm, y_denorm))

def _discover_parquet_files(input_dir, ship_type):
    """
    Discover and return a list of parquet files for the specified ship type.
    """
    file_names = os.listdir(input_dir)
    ship_type_dirs = [input_dir + '/' + fn + '/' + ship_type for fn in file_names]
    parquet_files = []
    vessel_ids = []
    for path in ship_type_dirs:
        # Get all vessel directories
        vessel_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
        for vessel_id in vessel_dirs:
            vessel_path = os.path.join(path, vessel_id)
            segment_dirs = [d for d in os.listdir(vessel_path) if os.path.isdir(os.path.join(vessel_path, d))]
            
            for segment_dir in segment_dirs:
                segment_path = os.path.join(vessel_path, segment_dir)
                segment_files = glob.glob(os.path.join(segment_path, "*.parquet"))
                parquet_files.extend(segment_files)
                vessel_ids.extend([vessel_id] * len(segment_files))
    print(f"Discovered {len(parquet_files)} parquet files from {len(set(vessel_ids))} vessels.")
    return parquet_files

def _compute_and_store_norm_stats(train_trajectories, json_dir):
    """
    Compute and store normalization statistics from the training dataset.
    """
    feature_list = ["X", "Y", "Delta_X", "Delta_Y", "SOG", "COG_sin", "COG_cos"]
    # check if all features are present
    if all(feature in train_trajectories[0].columns for feature in feature_list):
        all_data = np.concatenate([traj[feature_list].values for traj in train_trajectories], axis=0)
        train_means = np.mean(all_data, axis=0)
        train_stds = np.std(all_data, axis=0)    
    else:
        raise ValueError("Not all required features are present in the training trajectories.")
    #save training means and stds to file
    filename = os.path.join(json_dir, "train_data_statistics.json")
    with open(filename, "w") as f:
        json.dump({
            "train_feature_means": train_means.tolist(),
            "train_feature_stds": train_stds.tolist()
        }, f)
    return

def _resample_trajectory(df, freq=FREQUENCY, min_points=30, max_gap_seconds=3600):
    """
    Resample trajectory DataFrame to fixed frequency.
    """
    if len(df) < min_points:
        return None
    
    df = df.copy()
    df.set_index('Timestamp')

    # Ensure 'Timestamp' is datetime and set it as the index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()

    # Compute timedeltas between consecutive points
    time_diffs = df.index.to_series().diff()

# Convert max_gap to timedelta and get split points
    max_gap = pd.Timedelta(seconds=max_gap_seconds)
    split_points = time_diffs[time_diffs > max_gap].index

    
    if len(split_points) == 0:
        # No large gaps, process as single segment
        return _resample_single_segment(df, freq, min_points)
    else:
        # Split into multiple segments at large gaps
        segments = []
        start_idx = df.index[0]
        
        for split_point in split_points:
            segment_df = df.loc[start_idx:split_point - pd.Timedelta(seconds=1)]
            resampled_segment = _resample_single_segment(segment_df, freq, min_points)
            if resampled_segment is not None:
                segments.append(resampled_segment)
            start_idx = split_point
        
        # Process final segment
        final_segment = df.loc[start_idx:]
        resampled_final = _resample_single_segment(final_segment, freq, min_points)
        if resampled_final is not None:
            segments.append(resampled_final)
        
        print(f"  Split into {len(segments)} segments due to large gaps")
        return segments
    
def _resample_single_segment(df, freq, min_points):
    """Helper function to resample a single continuous segment"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not numeric_columns:
        return None
    
    # Resample only numeric columns with mean
    resampled_numeric = df[numeric_columns].resample(str(freq)+'s').mean()
    
    # For non-numeric columns, use first()
    if non_numeric_columns:
        resampled_non_numeric = df[non_numeric_columns].resample(str(freq)+'s').first()
        resampled_df = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
    else:
        resampled_df = resampled_numeric
    
    # Remove any NaN rows (from resampling periods with no data)
    resampled_df = resampled_df.dropna()
    
    if len(resampled_df) < min_points:
        return None
    
    resampled_df = resampled_df.reset_index()
    # drop the 'Timestamp' column after resampling
    resampled_df = resampled_df.drop(columns=['Timestamp'])
    return resampled_df

def _convert_latlon_to_xy(df, ref_lat, ref_lon):
    """
    Convert Latitude and Longitude to X and Y coordinates in meters.
    """
    R = 6371000  # Earth radius in meters
    lat_rad = np.deg2rad(df['Latitude'].values)
    lon_rad = np.deg2rad(df['Longitude'].values)
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = R * (lat_rad - ref_lat_rad)

    df = df.copy()
    df['X'] = x
    df['Y'] = y
    df = df.drop(columns=['Latitude', 'Longitude'])
    return df

def load_norm_stats(filepath):
    """
    Load normalization statistics from file.
    """
    with open(filepath, "r") as f:
        stats = json.load(f)
    train_means = np.array(stats["train_feature_means"])
    train_stds = np.array(stats["train_feature_stds"])
    return train_means, train_stds

def _apply_normalization(trajectories, normalization_path):
    """
    Apply normalization to the trajectories using stored statistics.
    """
    filepath = os.path.join(normalization_path, "train_data_statistics.json")
    train_means, train_stds = load_norm_stats(filepath)
    
    normalized_trajectories = []
    for df in trajectories:
        df_norm = df.copy()
        feature_list = ["X", "Y", "Delta_X", "Delta_Y", "SOG", "COG_sin", "COG_cos"]
        for i, feature in enumerate(feature_list):
            df_norm[feature] = (df_norm[feature] - train_means[i]) / train_stds[i]
        normalized_trajectories.append(df_norm)
    
    return normalized_trajectories

def _create_subsequences(trajectories, seq_length=30):
    """
    Create subsequences of fixed length from trajectories.
    """
    sequences = []
    for df in trajectories:
        data = df.values
        num_points = data.shape[0]
        if num_points < seq_length:
            continue
        for start_idx in range(num_points - seq_length + 1):
            subseq = data[start_idx:start_idx + seq_length]
            sequences.append(subseq)
    return sequences

class AISDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: list of numpy arrays, each shape (30, 7) [x, y, △x, △y, sog, cog_sin, cog_cos]
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Load raw sequence (shape: 30 × 7)
        seq = self.sequences[idx].astype(np.float32).copy()

        # Input: first 29 points
        input_seq = seq[:-1]  # shape: (29, 7)

        # Target: x, y of the final point
        targets = seq[-1, :2]  # shape: (2,)

        return input_seq, targets

    # main function for testing
if __name__ == "__main__":
    parquet_files = create_datasets('data/processed', 'Cargo')