import os
import time
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Get dataset path from .env
dataset_path = os.getenv("DATASET_PATH")

# Load the dataset
logging.info("Loading the dataset...")
df = pd.read_csv(dataset_path)

# Filter out top 1% and bottom 1% from the data
logging.info("Removing outliers...")
q_low = df['amount'].quantile(0.01)
q_high = df['amount'].quantile(0.99)
df = df[(df['amount'] >= q_low) & (df['amount'] <= q_high)]

# Drop unnecessary columns
logging.info("Dropping columns...")
df = df.drop(['step', 'initiator','recipient', 'isFraud'], axis=1)

# One-Hot Encoding (transactionType)
df = pd.get_dummies(df, columns=['transactionType'], drop_first=True)

# Setting features to scale
features_to_scale = [
    'amount',
    'oldBalInitiator',
    'newBalInitiator',
    'oldBalRecipient',
    'newBalRecipient',
]

binary_features = [
    'transactionType_DEPOSIT',
    'transactionType_PAYMENT',
    'transactionType_TRANSFER',
    'transactionType_WITHDRAWAL'
]

# Standardise the features specified
logging.info("Scaling features...")
scaler = StandardScaler()
scaled_continuous = scaler.fit_transform(df[features_to_scale])

# Convert the scaled data back to a dataframe
scaled_data = np.hstack([scaled_continuous, df[binary_features].values])
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# Convert the scaled data to CSV
logging.info("Feature scaling complete. Generating scaled data CSV...")
start_time = time.time()
scaled_df.to_csv('data/scaled_data.csv', index=False)
end_time = time.time() - start_time
logging.info(f"Scaled data CSV generated in {end_time:.3f} seconds.")
