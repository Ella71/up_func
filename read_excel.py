
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

original = pd.read_excel('data/original.xlsx')

def extract_lstm_features(df, LSTM_FEAT_SIZE=16, timesteps=10):
    co = []
    for i in df.columns:
        if i.startswith("preD_") or i.startswith("preTurn") or i.startswith("OPENAT") or i.startswith("preOpen_"):
            co.append(i)
    # co = ['preD_0', .. 'preD_13', 'OPENAT', 'preOpen_0', ... 'preOpen_13', 'preTurn0', ... 'preTurn13', 'preD_vol5', ... 'preD_vol1', 'preD_vol-1']
    class LSTMFeatureExtractor(nn.Module):
        def __init__(self, input_size, hidden_size=LSTM_FEAT_SIZE, num_layers=1):
            super(LSTMFeatureExtractor, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            out, (h_n, _) = self.lstm(x)
            # # 使用最后一个时间步的输出作为特征
            # return h_n[-1]  # shape: (batch, hidden_size)
            #  取最后一个时间步的 hidden state 作为特征
            return out[:, -1, :]  # shape: (batch_size, hidden_size)

    data = df[co].values
    input1 = data[:, co.index('preD_0'):co.index('preD_13')+1].reshape((data.shape[0], 14, 1))
    input2 = data[:, (co.index('OPENAT')):co.index('preOpen_13')+1].reshape((data.shape[0], 15, 1))
    input3 = data[:, co.index('preTurn0'):co.index('preTurn13')+1].reshape((data.shape[0], 14, 1))
    input4 = data[:, co.index('preD_vol5'):co.index('preD_vol-1')+1].reshape((data.shape[0], 6, 1))
    # Standardize features
    scaler = StandardScaler()
    input1 = scaler.fit_transform(input1)
    input2 = scaler.fit_transform(input2)
    input3 = scaler.fit_transform(input3)
    input4 = scaler.fit_transform(input4)
    
    input1 = torch.tensor(input1, dtype=torch.float32)
    input2 = torch.tensor(input2, dtype=torch.float32)
    input3 = torch.tensor(input3, dtype=torch.float32)
    input4 = torch.tensor(input4, dtype=torch.float32)

    lstm1 = LSTMFeatureExtractor(input_size=1)
    lstm2 = LSTMFeatureExtractor(input_size=1)
    lstm3 = LSTMFeatureExtractor(input_size=1)
    lstm4 = LSTMFeatureExtractor(input_size=1)
    lstm1.eval()
    lstm2.eval()
    lstm3.eval()
    lstm4.eval()

    # 提取特征
    feat1 = lstm1(input1)  # shape: (1, LSTM_FEAT_SIZE)
    feat2 = lstm2(input2)
    feat3 = lstm3(input3)
    feat4 = lstm4(input4)

    # 拼接特征
    combined_features = torch.cat([feat1, feat2, feat3, feat4], dim=1)
    print("拼接后的特征维度:", combined_features.shape)

    feature_df = pd.DataFrame(combined_features.detach().clone().cpu().numpy(), columns=[f'feat_{i}' for i in range(combined_features.shape[1])])
    feature_df


def main():
    print(extract_lstm_features(original))
    # features_df = extract_lstm_features(original)
    # print(features_df.head())

if __name__ == '__main__':
    main()
