import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

   
    X = df.drop('0', axis=1).values
    y = df['0'].values

   
    X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, X_test, y_train, y_test
