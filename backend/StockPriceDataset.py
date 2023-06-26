import torch
from torch.utils.data import Dataset

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset class
class StockPriceDataset(Dataset):
    def __init__(self, data, window_size, output_dim):
        self.window_size = window_size
        self.data = data
        self.output_dim = output_dim
        self.normalize()

    def normalize(self):
        self.data["daily_return"] = (self.data["daily_return"] - self.data["daily_return"].mean()) / self.data["daily_return"].std()

    def __len__(self):
        return len(self.data) - self.window_size + 1 - self.output_dim

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size -1].values # sequence of window_size - 1 days
        y = self.data.iloc[idx + self.window_size -1 : idx + self.window_size -1 + self.output_dim, -1:].values # next output_dim days
        return torch.tensor(x).float().to(device), torch.tensor(y).float().to(device)