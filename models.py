import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from LSTM import *

DAYS_TO_PREDICT = 21

def get_lstm(all_data):
    
    scaler = MinMaxScaler()
    scaler.fit(all_data.values.reshape(-1,1))
    data = scaler.transform(all_data.values.reshape(-1,1))

    seq_length = 10
    X_all, y_all = create_sequences(data, seq_length)

    X_all = torch.from_numpy(X_all).float()
    y_all = torch.from_numpy(y_all).float()

    model = COVIDCasePredictor(n_features=1, n_hidden=512, seq_len=seq_length, n_layers=2)
    model = train_model(model, X_all, y_all)

    with torch.no_grad():
        test_seq = X_all[:1]
        preds = []
        for _ in range(DAYS_TO_PREDICT):
            y_test_pred = model(test_seq)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

    predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

    return predicted_cases.tolist()

def get_all_preds(df, county):
    result = pd.DataFrame()
    result['Date'] = pd.date_range(
        start=df['Date'].iloc[-1],
        periods=DAYS_TO_PREDICT + 1,
        closed=None
    )

    diff = df[county].diff().fillna(df[county])
    predicted_cases = get_lstm(diff)
    predicted_cases = np.cumsum(predicted_cases)
    predicted_cases = [x + df[county].iloc[-1] for x in predicted_cases]
    predicted_cases.insert(0, df[county].iloc[-1])
    result['LSTM'] = predicted_cases

    return result
