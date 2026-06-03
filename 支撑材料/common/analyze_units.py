import pandas as pd
import numpy as np
import math
import sys

def read_csv_gbk(filepath):
    """Read CSV file with GBK encoding"""
    try:
        df = pd.read_csv(filepath, encoding='gbk')
        return df
    except:
        # Try other encodings if GBK fails
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            return df
        except:
            df = pd.read_csv(filepath, encoding='latin1')
            return df

def main():
    # Read data files
    users_path = r'C:\Users\John\Desktop\国赛校\重庆大学2026数学建模春季赛题目\B题附件\users.csv'
    spots_path = r'C:\Users\John\Desktop\国赛校\重庆大学2026数学建模春季赛题目\B题附件\spots.csv'
    
    print("Reading users.csv...")
    users_df = read_csv_gbk(users_path)
    print(f"Users shape: {users_df.shape}")
    print(f"Users columns: {list(users_df.columns)}")
    
    print("\nReading spots.csv...")
    spots_df = read_csv_gbk(spots_path)
    print(f"Spots shape: {spots_df.shape}")
    print(f"Spots columns: {list(spots_df.columns)}")
    
    # Rename columns for clarity (based on garbled Chinese headers)
    # users.csv: 用户ID, 目的地X, 目的地Y, 最大步行距离, 到达时间, 离开时间
    # spots.csv: 车位ID, X坐标, Y坐标
    
    if len(users_df.columns) == 6:
        users_df.columns = ['user_id', 'dest_x', 'dest_y', 'max_walk_dist', 'arrival_time', 'departure_time']
    else:
        # If column names are garbled, assume first 6 columns
        users_df = users_df.iloc[:, :6]
        users_df.columns = ['user_id', 'dest_x', 'dest_y', 'max_walk_dist', 'arrival_time', 'departure_time']
    
    if len(spots_df.columns) == 3:
        spots_df.columns = ['spot_id', 'spot_x', 'spot_y']
    else:
        # If column names are garbled, assume first 3 columns
        spots_df = spots_df.iloc[:, :3]
        spots_df.columns = ['spot_id', 'spot_x', 'spot_y']
    
    # Convert to numeric
    users_df['dest_x'] = pd.to_numeric(users_df['dest_x'], errors='coerce')
    users_df['dest_y'] = pd.to_numeric(users_df['dest_y'], errors='coerce')
    users_df['max_walk_dist'] = pd.to_numeric(users_df['max_walk_dist'], errors='coerce')
    
    spots_df['spot_x'] = pd.to_numeric(spots_df['spot_x'], errors='coerce')
    spots_df['spot_y'] = pd.to_numeric(spots_df['spot_y'], errors='coerce')
    
    # Basic statistics
    print("\n=== USER COORDINATE STATISTICS ===")
    print(f"Number of users: {len(users_df)}")
    print(f"Destination X - Min: {users_df['dest_x'].min():.2f}, Max: {users_df['dest_x'].max():.2f}, Mean: {users_df['dest_x'].mean():.2f}, Std: {users_df['dest_x'].std():.2f}")
    print(f"Destination Y - Min: {users_df['dest_y'].min():.2f}, Max: {users_df['dest_y'].max():.2f}, Mean: {users_df['dest_y'].mean():.2f}, Std: {users_df['dest_y'].std():.2f}")
    print(f"Max walk distance - Min: {users_df['max_walk_dist'].min():.2f}, Max: {users_df['max_walk_dist'].max():.2f}, Mean: {users_df['max_walk_dist'].mean():.2f}, Std: {users_df['max_walk_dist'].std():.2f}")
    
    print("\n=== SPOT COORDINATE STATISTICS ===")
    print(f"Number of spots: {len(spots_df)}")
    print(f"Spot X - Min: {spots_df['spot_x'].min():.2f}, Max: {spots_df['spot_x'].max():.2f}, Mean: {spots_df['spot_x'].mean():.2f}, Std: {spots_df['spot_x'].std():.2f}")
    print(f"Spot Y - Min: {spots_df['spot_y'].min():.2f}, Max: {spots_df['spot_y'].max():.2f}, Mean: {spots_df['spot_y'].mean():.2f}, Std: {spots_df['spot_y'].std():.2f}")
    
    # Compute Euclidean distances for all user-spot pairs
    print("\n=== COMPUTING ALL USER-SPOT DISTANCES ===")
    n_users = len(users_df)
    n_spots = len(spots_df)
    total_pairs = n_users * n_spots
    print(f"Computing {total_pairs:,} distances ({n_users} users × {n_spots} spots)...")
    
    # Use vectorized computation for efficiency
    distances = []
    
    # For large datasets, process in batches
    batch_size = 1000
    for i in range(0, n_users, batch_size):
        user_batch = users_df.iloc[i:i+batch_size]
        user_x = user_batch['dest_x'].values
        user_y = user_batch['dest_y'].values
        
        # Compute distances to all spots for this batch
        for j in range(n_spots):
            spot_x = spots_df.iloc[j]['spot_x']
            spot_y = spots_df.iloc[j]['spot_y']
            batch_dist = np.sqrt((user_x - spot_x)**2 + (user_y - spot_y)**2)
            distances.extend(batch_dist)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, n_users)}/{n_users} users...")
    
    distances = np.array(distances)
    
    print(f"\n=== DISTANCE DISTRIBUTION ===")
    print(f"Total pairs: {len(distances):,}")
    print(f"Min distance: {distances.min():.4f}")
    print(f"Max distance: {distances.max():.4f}")
    print(f"Mean distance: {distances.mean():.4f}")
    print(f"Median distance: {np.median(distances):.4f}")
    print(f"Std deviation: {distances.std():.4f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(distances, p):.4f}")
    
    # Count pairs with distance <= 15.33 (claimed average walk threshold w_j)
    threshold = 15.33
    within_threshold = np.sum(distances <= threshold)
    percentage = (within_threshold / len(distances)) * 100
    print(f"\nPairs with distance <= {threshold}: {within_threshold:,} ({percentage:.2f}%)")
    
    # Theoretical average distance in a 100x100 unit square
    # For uniformly distributed points in [0,100]×[0,100]
    # Expected Euclidean distance ≈ 52.34 (theoretical value for unit square [0,1]×[0,1] is ~0.5214, scaled by 100)
    theoretical_avg = 0.5214 * 100  # For unit square scaled by 100
    print(f"\n=== THEORETICAL COMPARISON ===")
    print(f"Theoretical average distance in [0,100]×[0,100] unit square: ~{theoretical_avg:.2f}")
    print(f"Actual average distance: {distances.mean():.2f}")
    print(f"Ratio (actual/theoretical): {distances.mean()/theoretical_avg:.3f}")
    
    # Check if coordinates could be in meters
    print(f"\n=== PHYSICAL INTERPRETATION ===")
    print(f"If coordinates are in meters in a 100m×100m area:")
    print(f"  - Average walking distance would be {distances.mean():.2f} meters")
    print(f"  - This is {distances.mean():.2f} meters ≈ {distances.mean()/1000:.2f} km")
    print(f"  - Paper claims average walking distance of 2.82 units")
    print(f"  - Ratio: 2.82 / {distances.mean():.2f} = {2.82/distances.mean():.4f}")
    
    # Alternative: if coordinates are in 100m units (hectometers)
    print(f"\nIf coordinates are in 100m units (hectometers):")
    print(f"  - Actual distance in meters = {distances.mean() * 100:.2f} meters")
    print(f"  - Ratio to paper's 2.82: {distances.mean() * 100 / 2.82:.2f}")
    
    # Alternative: if coordinates are in km
    print(f"\nIf coordinates are in km:")
    print(f"  - Actual distance in meters = {distances.mean() * 1000:.2f} meters")
    print(f"  - Ratio to paper's 2.82: {distances.mean() * 1000 / 2.82:.2f}")
    
    return users_df, spots_df, distances

if __name__ == "__main__":
    users_df, spots_df, distances = main()