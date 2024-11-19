import torch
from torch.utils.data import Dataset, DataLoader

class P2S_Dataset(Dataset):    
    def __init__(self, data, labels):
        self.data = data  # Assume data is a dictionary with multiple columns
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract only specific columns
        input_data = self.data[idx]
        label = self.labels[idx]
        return input_data, label


"""
class P2S_Data(Dataset):
    def __init__(self):
        dataset = load_dataset('AIML-TUDA/P2S', 'Normal', download_mode='reuse_dataset_if_exists')

        # Extracting the time series data from the train and test dataset 
        ts_train = dataset['train']['dowel_deep_drawing_ow']
        ts_test = dataset['test']['dowel_deep_drawing_ow']

        ts_train = np.array(ts_train)
        ts_test = np.array(ts_test)

        # Adding a third dimension, in the middle of the shape, as needed for SAX
        ts_train = ts_train.reshape(ts_train.shape[0], 1, ts_train.shape[1])
        ts_test = ts_test.reshape(ts_test.shape[0], 1, ts_test.shape[1])



        self.data = torch.randn(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 images
        self.labels = torch.randint(0, 2, (100,))  # Binary labels for classification

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

# Create dataset and DataLoader
dataset = P2S_Data()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate through DataLoader
for batch_data, batch_labels in dataloader:
    print(batch_data.shape)  # e.g., torch.Size([16, 3, 32, 32])
    print(batch_labels.shape)  # e.g., torch.Size([16])
"""