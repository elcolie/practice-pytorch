import torch
from torch.utils.data import Dataset

HEADERS = ["Pclass", "Sex", "Age", "Parch", "Fare", "Embarked", "Name_length", "Has_Cabin", "FamilySize", "IsAlone",
           "Title"]
TARGET = ['Survived']


class TitanicDataset(Dataset):
    def __init__(self, df):
        self.x_data = df[HEADERS]
        self.y_data = df[TARGET]
        self.len = self.x_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.tensor(self.x_data.iloc[index].values)
        y = torch.tensor(self.y_data.iloc[index].values)
        return x, y
