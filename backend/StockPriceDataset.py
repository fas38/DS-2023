import torch
from torch.utils.data import Dataset
import pandas as pd

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset class
class StockPriceDataset(Dataset):
    def __init__(self, data, window_size, output_dim):
        self.window_size = window_size
        self.data = data
        self.output_dim = output_dim
        self.normalize()
        self.sequenced_data = self.sequence_data()

    def normalize(self):
        
        # normalize data
        self.data["daily_return"] = (self.data["daily_return"] - self.data["daily_return"].min()) / (self.data["daily_return"].max() - self.data["daily_return"].min() + 1e-8)
        self.data["sentiment"] = (self.data["sentiment"] - self.data["sentiment"].min()) / (self.data["sentiment"].max() - self.data["sentiment"].min() + 1e-8)
        self.data["sentiment_company"] = (self.data["sentiment_company"] - self.data["sentiment_company"].min()) / (self.data["sentiment_company"].max() - self.data["sentiment_company"].min() + 1e-8)
        self.data["sentiment_industry"] = (self.data["sentiment_industry"] - self.data["sentiment_industry"].min()) / (self.data["sentiment_industry"].max() - self.data["sentiment_industry"].min() + 1e-8)

    # create sequences by seperating each month data into seperate sequence
    def sequence_data(self):
        data = self.data
        list_of_df = []
        temp = data.copy()
        temp = temp.reset_index()
        temp["Date"] = pd.to_datetime(temp["Date"])
        all_available_dates = temp["Date"].unique()
    
        for i in range(len(all_available_dates)):
            date = all_available_dates[i]
            temp_df = data[data.index == date]
            list_of_df.append(temp_df)
        return list_of_df
        

    def __len__(self):
        return len(self.sequenced_data) - self.window_size - self.output_dim + 1

    def __getitem__(self, idx):
        if self.window_size == 1:
            sequences = self.sequenced_data[idx]
            data = sequences
        else:
            sequences = self.sequenced_data[idx: idx + self.window_size]
            data = pd.concat(sequences, axis=0)
            # data = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq.values).float() for seq in sequences], batch_first=True)
        x = data.iloc[:, :].values
        # x = data
        if self.window_size == 1:
            sequences_output = self.sequenced_data[idx: idx + self.output_dim]
            data_output = sequences_output[0]
        else:
            sequences_output = self.sequenced_data[idx : idx + self.window_size + self.output_dim]
            data_output = pd.concat(sequences_output, axis=0)
        y = data_output.iloc[-self.output_dim:, -1:].values
        return torch.tensor(x).float().to(device), torch.tensor(y).float().to(device)
        # return x.to(device), torch.tensor(y).float().to(device)
    
    def return_sequences(self, idx):
        return self.sequenced_data[idx: idx+1]