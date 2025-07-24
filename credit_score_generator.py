import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import os
import sys # Added for command-line arguments

# --- Feature Engineering Function (from Cell 5) ---
def engineer_wallet_features(transactions_df):
    # Rename 'userWallet' to 'walletaddress' (Corrected: 'userWallet' with uppercase 'W')
    if 'userWallet' in transactions_df.columns:
        transactions_df.rename(columns={'userWallet': 'walletaddress'}, inplace=True)
    # If 'walletaddress' is still not found after checking for 'userWallet',
    # and if 'from' and 'to' are present, we might need a more sophisticated way
    # to derive 'walletaddress' for feature engineering,
    # or ensure the input JSON always has 'userWallet' or 'walletaddress'.
    # Given the sample JSON, 'userWallet' is the expected key.
    elif 'walletaddress' not in transactions_df.columns and 'from' in transactions_df.columns:
        # Fallback: if userWallet isn't there, and 'from' is, use 'from' as walletaddress
        # This assumes each transaction's 'from' is the wallet of interest for that row.
        # This is less ideal than explicit 'userWallet' but can prevent KeyErrors
        # if the input format changes unexpectedly.
        transactions_df['walletaddress'] = transactions_df['from']
        print("Warning: 'userWallet' not found, using 'from' column as 'walletaddress'.")


    # Convert timestamp to datetime and extract date
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'], unit='ms')
    transactions_df['date'] = transactions_df['timestamp'].dt.date

    # Extract 'amount' and 'assetSymbol' from 'actionData'
    # Safely extract 'amount' and 'assetSymbol'
    transactions_df['amount_extracted'] = transactions_df['actionData'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) and 'amount' in x else None
    )
    transactions_df['token_extracted'] = transactions_df['actionData'].apply(
        lambda x: x.get('assetSymbol') if isinstance(x, dict) and 'assetSymbol' in x else 'UNKNOWN'
    )

    # Convert amount to numeric, handling potential errors
    # Use errors='coerce' to turn unparseable values into NaN, then fill NaN
    transactions_df['amount_numeric'] = pd.to_numeric(transactions_df['amount_extracted'], errors='coerce')
    # FIX: Addressed FutureWarning for inplace=True
    transactions_df['amount_numeric'] = transactions_df['amount_numeric'].fillna(0) # Fill NaNs with 0

    # --- Aggregate Features by walletAddress ---
    # Ensure 'walletaddress' exists before grouping. If not, raise a clear error.
    if 'walletaddress' not in transactions_df.columns:
        raise KeyError("After initial processing, 'walletaddress' column is still missing. "
                       "Ensure your input JSON has 'userWallet' or 'walletaddress' field.")

    wallet_features = transactions_df.groupby('walletaddress').agg(
        total_transactions=('action', 'count'),
        active_days=('date', lambda x: x.nunique()),
        duration_days=('date', lambda x: (x.max() - x.min()).days + 1 if x.nunique() > 1 else 1),
        first_transaction_date=('timestamp', 'min'),
        last_transaction_date=('timestamp', 'max'),
        unique_tokens_interacted=('token_extracted', lambda x: x.nunique())
    )

    # Calculate transactions_per_day
    wallet_features['transactions_per_day'] = wallet_features['total_transactions'] / wallet_features['duration_days']
    wallet_features['transactions_per_day'].replace([np.inf, -np.inf], 0, inplace=True) # Handle division by zero duration

    # Calculate avg_time_between_tx_hours (if more than 1 transaction)
    def calculate_avg_time_between_tx(timestamps):
        if len(timestamps) > 1:
            timestamps = timestamps.sort_values()
            time_diffs = (timestamps - timestamps.shift(1)).dropna()
            return time_diffs.mean().total_seconds() / 3600 # in hours
        return 0.0

    avg_time_between_tx = transactions_df.groupby('walletaddress')['timestamp'].apply(calculate_avg_time_between_tx)
    wallet_features['avg_time_between_tx_hours'] = avg_time_between_tx.reindex(wallet_features.index).fillna(0)


    # Calculate last_transaction_recency_days (relative to the latest transaction in the *entire dataset*)
    # This makes recency comparable across all wallets in the input.
    latest_overall_transaction = transactions_df['timestamp'].max()
    wallet_features['last_transaction_recency_days'] = (latest_overall_transaction - wallet_features['last_transaction_date']).dt.days

    # Aggregate financial action amounts and counts
    action_pivot = transactions_df.pivot_table(
        index='walletaddress',
        columns='action',
        values='amount_numeric',
        aggfunc=['sum', 'count']
    ).fillna(0)

    # Flatten multi-level columns
    action_pivot.columns = ['_'.join(col).strip() for col in action_pivot.columns.values]

    # Select and rename relevant columns, add if they don't exist
    financial_cols = {
        'sum_deposit': 'total_deposit_amount',
        'sum_borrow': 'total_borrow_amount',
        'sum_repay': 'total_repay_amount',
        'sum_redeemunderlying': 'total_redeem_amount',
        'sum_liquidationcall': 'total_liquidation_call_amount',
        'count_deposit': 'count_deposit',
        'count_borrow': 'count_borrow',
        'count_repay': 'count_repay',
        'count_redeemunderlying': 'count_redeemunderlying',
        'count_liquidationcall': 'count_liquidationcall'
    }

    for old_col, new_col in financial_cols.items():
        if old_col in action_pivot.columns:
            wallet_features[new_col] = action_pivot[old_col]
        else:
            wallet_features[new_col] = 0.0 # Add as 0 if action didn't occur for any wallet

    # Calculate average amounts for each action type
    # Using np.divide and np.where to handle division by zero more cleanly
    wallet_features['avg_deposit_amount'] = np.where(wallet_features['count_deposit'] > 0, wallet_features['total_deposit_amount'] / wallet_features['count_deposit'], 0)
    wallet_features['avg_borrow_amount'] = np.where(wallet_features['count_borrow'] > 0, wallet_features['total_borrow_amount'] / wallet_features['count_borrow'], 0)
    wallet_features['avg_repay_amount'] = np.where(wallet_features['count_repay'] > 0, wallet_features['total_repay_amount'] / wallet_features['count_repay'], 0)
    wallet_features['avg_redeem_amount'] = np.where(wallet_features['count_redeemunderlying'] > 0, wallet_features['total_redeem_amount'] / wallet_features['count_redeemunderlying'], 0)


    # Fill potential NaNs from division by zero counts
    # The np.where already handles this, but keep for robustness against other NaNs
    wallet_features.fillna(0, inplace=True)
    wallet_features.replace([np.inf, -np.inf], 0, inplace=True)

    # Behavioral Ratios
    # Avoid division by zero by adding a small epsilon or checking for zero
    epsilon = 1e-9 # A small number to avoid division by zero

    wallet_features['borrow_to_deposit_ratio'] = wallet_features['total_borrow_amount'] / (wallet_features['total_deposit_amount'] + epsilon)
    wallet_features['repay_to_borrow_ratio'] = wallet_features['total_repay_amount'] / (wallet_features['total_borrow_amount'] + epsilon)
    wallet_features['redeem_to_deposit_ratio'] = wallet_features['total_redeem_amount'] / (wallet_features['total_deposit_amount'] + epsilon)

    # Net Borrow-Repay
    wallet_features['net_borrow_repay'] = wallet_features['total_borrow_amount'] - wallet_features['total_repay_amount']

    # Standard deviations of amounts (if more than 1 transaction of that type)
    def safe_std(series):
        return series.std() if len(series) > 1 else 0.0

    std_amounts = transactions_df.groupby(['walletaddress', 'action'])['amount_numeric'].apply(safe_std).unstack(fill_value=0)
    for col in ['deposit', 'borrow', 'repay']:
        if col in std_amounts.columns:
            wallet_features[f'std_{col}_amount'] = std_amounts[col]
        else:
            wallet_features[f'std_{col}_amount'] = 0.0

    # Clean up temporary columns and dates
    wallet_features.drop(columns=['first_transaction_date', 'last_transaction_date'], inplace=True)

    # Ensure all columns are numeric
    for col in wallet_features.columns:
        if wallet_features[col].dtype == 'object':
            wallet_features[col] = pd.to_numeric(wallet_features[col], errors='coerce').fillna(0)

    return wallet_features


# --- The One-Step Scoring Function (from Cell 8) ---
def generate_wallet_scores_from_json(json_file_path, models_dir='models/'):
    """
    Generates wallet credit scores from a JSON file of transactions using pre-trained models.

    Args:
        json_file_path (str): Path to the input JSON transaction file.
        models_dir (str): Directory where the trained model artifacts are saved.

    Returns:
        dict: A dictionary where keys are wallet addresses and values are their credit scores.
              Returns an empty dictionary if any step fails.
    """
    # 1. Load Data
    print(f"Loading transaction data from {json_file_path}...")
    try:
        with open(json_file_path, 'r') as f:
            transactions_data = json.load(f)
        transactions_df_input = pd.DataFrame(transactions_data)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}

    if transactions_df_input.empty:
        print("Input transaction DataFrame is empty. No scores to generate.")
        return {}

    # --- CRITICAL FIX: Ensure 'walletaddress' column exists (Case-Sensitive fix) ---
    # This block was moved from engineer_wallet_features to here
    # to ensure 'walletaddress' is set *before* engineer_wallet_features is called.
    if 'walletaddress' not in transactions_df_input.columns:
        if 'userWallet' in transactions_df_input.columns: # Corrected: uppercase 'W'
            transactions_df_input.rename(columns={'userWallet': 'walletaddress'}, inplace=True)
            print("Renamed 'userWallet' to 'walletaddress'.")
        elif 'from' in transactions_df_input.columns: # Fallback if 'userWallet' is not present
            transactions_df_input['walletaddress'] = transactions_df_input['from']
            print("Warning: 'userWallet' not found in input data. Using 'from' column as 'walletaddress'. "
                  "Ensure this is the intended behavior for your data.")
        else:
            raise KeyError("Input JSON must contain 'walletaddress', 'userWallet', or 'from' column to identify wallets.")
    # --- END OF CRITICAL FIX ---

    # 2. Load Pre-trained Models
    try:
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        pca = joblib.load(os.path.join(models_dir, 'pca.pkl'))
        kmeans_model = joblib.load(os.path.join(models_dir, 'kmeans_model.pkl'))
        cluster_score_mapping = joblib.load(os.path.join(models_dir, 'cluster_score_mapping.pkl'))
        trained_feature_columns = joblib.load(os.path.join(models_dir, 'trained_feature_columns.pkl'))
        print("Pre-trained models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: One or more pre-trained models not found in {models_dir}. Missing: {e.filename}")
        print("Please ensure models are saved in the 'models/' directory by running Cell 6 in the notebook.")
        return {}
    except Exception as e:
        print(f"Error loading models: {e}")
        return {}

    # 3. Feature Engineering for the input data
    print("Engineering features for input wallets...")
    # Pass a copy to engineer_wallet_features to avoid modifying the original DataFrame
    # if it's used elsewhere, though 'copy()' is already done by the caller.
    wallet_features_input_df = engineer_wallet_features(transactions_df_input.copy())

    if wallet_features_input_df.empty:
        print("Feature engineering resulted in an empty DataFrame. Cannot proceed with scoring.")
        return {}

    # 4. Align features to match training data's columns and order
    missing_cols_in_input = set(trained_feature_columns) - set(wallet_features_input_df.columns)
    for c in missing_cols_in_input:
        wallet_features_input_df[c] = 0.0 # Add missing columns with default 0.0 value

    extra_cols_in_input = set(wallet_features_input_df.columns) - set(trained_feature_columns)
    if extra_cols_in_input:
        wallet_features_input_df = wallet_features_input_df.drop(columns=list(extra_cols_in_input))
        print(f"Dropped extra columns from input features: {list(extra_cols_in_input)}")

    try:
        wallet_features_input_df = wallet_features_input_df[trained_feature_columns]
    except KeyError as e:
        print(f"Error: Mismatch in expected feature columns. Missing column: {e}. Check `trained_feature_columns.pkl` and input data.")
        return {}

    # 5. Preprocess Features (Scaling and PCA)
    print("Preprocessing features...")
    if wallet_features_input_df.empty or wallet_features_input_df.shape[1] == 0:
        print("No features to preprocess for input wallets. Skipping prediction.")
        return {}

    wallet_features_input_df.fillna(0, inplace=True)
    wallet_features_input_df.replace([np.inf, -np.inf], 0, inplace=True)

    scaled_features_input = scaler.transform(wallet_features_input_df)

    if pca.n_components_ == 0:
        print("Warning: PCA resulted in 0 components. Assigning neutral scores.")
        final_scores = {wallet_address: 500 for wallet_address in wallet_features_input_df.index}
        return final_scores
    elif pca.n_components_ == 1:
        pca_features_input = pca.transform(scaled_features_input).reshape(-1, 1)
    else:
        pca_features_input = pca.transform(scaled_features_input)

    # 6. Predict Clusters
    print("Predicting clusters for wallets...")
    if pca_features_input.shape[0] == 0:
        print("No PCA features to predict. Skipping cluster prediction.")
        return {}

    if kmeans_model is None:
        print("K-Means model not available (e.g., only 1 cluster in training). Assigning neutral scores.")
        final_scores = {wallet_address: 500 for wallet_address in wallet_features_input_df.index}
        return final_scores
    else:
        cluster_labels_input = kmeans_model.predict(pca_features_input)

    # 7. Assign Credit Scores
    print("Assigning credit scores...")
    final_scores = {}
    for i, wallet_address in enumerate(wallet_features_input_df.index):
        cluster_id = cluster_labels_input[i]
        score_info = cluster_score_mapping.get(cluster_id)
        if score_info:
            score = (score_info['min'] + score_info['max']) / 2
            final_scores[wallet_address] = int(round(score))
        else:
            final_scores[wallet_address] = 500

    print("Score generation complete.")
    return final_scores

# --- Command-line execution block ---
if __name__ == '__main__':
    # Expects the path to the input JSON file as a command-line argument
    if len(sys.argv) < 2:
        print('Usage: python credit_score_generator.py <path_to_input_json_file> [output_json_file_name]')
        sys.exit(1)

    input_json_path = sys.argv[1]
    # Default output file name is 'generated_wallet_scores.json'
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else 'generated_wallet_scores.json'

    print(f'Generating scores for {input_json_path}...')
    scores = generate_wallet_scores_from_json(input_json_path)

    if scores:
        with open(output_json_path, 'w') as f:
            json.dump(scores, f, indent=4)
        print(f'Scores saved to {output_json_path}')
    else:
        print('No scores were generated.')