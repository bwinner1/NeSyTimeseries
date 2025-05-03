import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from torch import nn


import datetime
import datasets
import matplotlib.pyplot as plt

from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_unit_test

from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

sys.path.append(os.path.abspath('VQShape'))

from einops import rearrange
from vqshape.pretrain import LitVQShape

from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler


############
# Transformers #
############
"""
Code largely from https://github.com/juho-lee/set_transformer by yoonholee.
"""
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    """
    Set Transformer used for the Neuro-Symbolic Concept Learner.
    """
    def __init__(self, dim_input=3, dim_output=40, dim_hidden=128, num_heads=4, ln=False):
        """
        Builds the Set Transformer.
        :param dim_input: Integer, input dimensions
        :param dim_output: Integer, output dimensions
        :param dim_hidden: Integer, hidden layer dimensions
        :param num_heads: Integer, number of attention heads
        :param ln: Boolean, whether to use Layer Norm
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_input, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
            SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=1, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze()


############
# Summarizers #
############

class SAXTransformer:
    def __init__(self, n_segments=8, alphabet_size=4):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.sax = SAX(self.n_segments, self.alphabet_size)

    def transform(self, dataset):

        # Add a third dimension in the middle of the shape, as needed for SAX
        dataset = dataset.reshape(dataset.shape[0], 1, dataset.shape[1])

        # Make sure that number of time steps is dividable by n_segments
        remainder = dataset.shape[2] % self.n_segments
        if(remainder != 0):
            fillers = self.n_segments - remainder
            meansToFill = np.repeat(np.mean(dataset, axis=2), fillers, axis=1)
            dataset = np.append(dataset[:,0,:], meansToFill, axis = 1) 
            # append function automatically removes all 1s in array shape, add it manually:  
            dataset = np.expand_dims(dataset, 1)
            print(f"SAX: Added {self.getEntity(fillers, 'mean')}")
            
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Fit the scaler on the training data (learn the mean and standard deviation)
        # Transform both the training and test data
        # Transposition on input and output added, as StandardScaler operates column-wise (e.g. calculates mean of first column)
        dataset_scaled = scaler.fit_transform(dataset[:, 0, :].T).T

        # Fit and transform the data
        dataset_sax = self.sax.fit_transform(dataset_scaled)

        # Remove the inner dimension from output
        dataset_sax = torch.squeeze(torch.tensor(dataset_sax))
        
        return dataset_sax

    @staticmethod
    def getEntity(number, name):
        correctName = name + "s" if number > 1 else name
        return f"{number} {correctName}"

    # Print general information of dataset
    def printGeneralInfo(self, dataset, dataset_scaled, dataset_sax, increment, time_steps):

        print(f"dataset.shape: {dataset.shape}")
        samples = self.getEntity(dataset.shape[0], 'sample')
        dimensions = self.getEntity(dataset.shape[1], 'dimension')
        time_steps = self.getEntity(dataset.shape[2], 'time step')

        print(f"{samples}, {dimensions}, {time_steps}\n")

        print(f"dataset_scaled.shape: {dataset_scaled.shape}")
        print(f"dataset_sax.shape: {dataset_sax.shape}")

        segments = self.getEntity(self.n_segments, "segment")
        increments = self.getEntity(increment, "time step")
        print(f"\n{segments}")
        print(f"{increments} per segment")


    # Print a grid of samples from a given dataset
    # rows and columns have to be greater than 1
    def plotGrid(self, dataset, dataset_scaled, dataset_sax, rows=2, columns=3, dataset_name="example_dataset"):

        # Define segments as pairs of (start, end) indices with constant values
        # segments = [(5, 20), (20, 30), (30, 50), (50, 70), (70, 80), (80, 95)]  # Segment boundaries
        # segment_values = [4, 6, 5, 7, 3, 8]  # Horizontal segment values for the plot (y-values)

        time_steps = dataset.shape[2]
        assert time_steps % self.n_segments == 0, "Number of time frames has to be evenly dividable by the number of segments."

        increment = int(time_steps / self.n_segments)

        self.printGeneralInfo(dataset, dataset_scaled, dataset_sax, increment, time_steps)

        # Define a Colormap
        get_color = colormaps['tab10']

        fig, grid = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
        for i, row in enumerate(grid):
            for j, ax in enumerate(row):
                sample_index = i * columns + j
                sample = dataset[sample_index][0]
                sample_sax = dataset_sax[sample_index][0]

                # Plot original time series
                ax.plot(sample)
                ax.set_title(f"Sample {sample_index + 1}")

                start = 0
                end = increment

                for s in range(self.n_segments):
                    s_value = np.mean(sample[start : end])

                    # Horizontal segment values for the plot (y-values)
                    color = get_color(sample_sax[s])
                    ax.hlines(y=s_value, xmin=start, xmax=end-1, color=color, linewidth=3)  # Horizontal line

                    # Update indices
                    start = end
                    end += increment

        plt.tight_layout()
        filename = "plot_" + dataset_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        plt.savefig(filename)
        plt.show()
    
class tsfreshTransformer:
    
    @staticmethod
    def fit_scaler(dataset):
        scaler = StandardScaler()
        scaler.fit(dataset)
        return scaler
        
    @staticmethod
    def transform(dataset, y, columns=None, scaler=None, setting="min", filter=True):
        # dataset, y = load_unit_test()
        dataset = np.array(dataset)
        # Probably no 
        # dataset = np.squeeze(dataset)
        y = pd.Series(y)

        # Create indices using np.arange and repeat them for each time series
        indices = np.repeat(np.arange(dataset.shape[0]), dataset.shape[1])

        # Flatten the dataset and create the DataFrame
        df = pd.DataFrame({
            "ts_id": indices,
            "ts": dataset.flatten()
        })

        # Possible Function Calculator Parameters:
        # ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

        fcparams = {"slow" : ComprehensiveFCParameters(),
                    "mid" : EfficientFCParameters(),
                    "fast" : MinimalFCParameters()}
        # Only extract
        X = extract_features(df, column_id='ts_id', impute_function=impute,
                              default_fc_parameters=fcparams[setting])
        
        """
        # Extract and filter
        # X = extract_relevant_features(df, y, column_id='ts_id',
        #                                default_fc_parameters=MinimalFCParameters())
        """
        
        # Train dataset
        #If no columns are given, then use select_features from tsfresh package
        if columns is None:
            if filter:
                X = select_features(X, y)            
            columns = X.columns
            scaler = StandardScaler()
            scaler.fit(X)

        # Val and Test datasets
        # If columns are given, then select the same for the train and test dataset
        else:
            X = X[columns]

        print("X.shape:")
        print(X.shape)
        
        X_normalized = X.to_numpy()
        
        # Convert the pd to a tensor, add an inner dimension
        return torch.tensor(X_normalized, dtype=torch.float32).unsqueeze(1), columns, scaler

class vqshapeTransformer:
    def __init__(self, model_to_use):

        ### Loading Checkpoint
        dims = (512, 256, 512)
        codes = (64, 512, 64)
        self.dim = dims[model_to_use]
        print("self.dim")
        print(self.dim)
        self.code = codes[model_to_use]
        checkpoint = f"uea_dim{self.dim}_codebook{self.code}"

        checkpoint_path = f"VQShape/checkpoints/vqshape_pretrain/{checkpoint}/VQShape.ckpt"
        lit_model = LitVQShape.load_from_checkpoint(checkpoint_path, 'cuda')
        self.model = lit_model.model

    def transform(self, dataset, mode = "tokenize"):

        x = torch.tensor(dataset, device='cuda', dtype=torch.float32)
        print("x.size()")
        print(x.size())

        # F.interpolate requires dimensions (batch, channels, time/height/width)
        x = x.unsqueeze(1)
        x = F.interpolate(x, self.dim, mode='linear')  # first interpolate to 512 timesteps
        print("Before inputting into model:") 
        print("x.size()")
        print(x.size())
        x = x.squeeze(1)
        print(x.size())

        if mode == 'evaluate':
            output_dict, loss_dict = self.model(x, mode='evaluate')
            return output_dict, loss_dict

        elif mode == 'tokenize':

            output = self.model(x, mode='tokenize')[0] # tokenize with VQShape
        
            histogramm = output['histogram'] 
            hist_of_hist = torch.sum(histogramm, 0)

            histogramm = histogramm.unsqueeze(1)
            return histogramm

    


############
# NeSyConceptLearner #
############
class NeSyConceptLearner(nn.Module):
    """
    The Neuro-Symbolic Concept Learner of Stammer et al. 2021 based on Slot Attention and Set Transformer.
    """
    
    def __init__(self, n_input_dim, n_set_heads, set_transf_hidden, n_classes = 2,
                 device='cuda'):       
        """
        :param n_classes: Integer, number of classes
        :param n_attr: Integer, number of attributes per object
        :param n_set_heads: Integer, number of attention heads for set transformer
        :param set_transf_hidden: Integer, hidden dim of set transformer
        prediction
        :param device: String, either 'cpu' or 'cuda'
        Note: set_transf_hidden % n_set_heads == 0 has to true!
        """
        
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_input_dim = n_input_dim
        
        # Concept Embedding Module
        # --- left out; is applied before the network ---

        # Reasoning module
        self.set_cls = SetTransformer(dim_input=n_input_dim, dim_hidden=set_transf_hidden, num_heads=n_set_heads,
                                      dim_output=n_classes, ln=True)

    def forward(self, attrs):
        """
        Receives a concept and passes it through the reasoning module.
        The ouputs of both modules are returned, i.e. the final class prediction and the latent symbolic representation.
        :param attrs: 3D Tensor[batch, sets, features]
        :return: Tuple of outputs of both modules, [batch, n_classes] classification/ reasoning module output,
        [batch, n_slots, n_attr] concept embedding module output/ symbolic representation
        """

        attrs = attrs.float()
        cls = self.set_cls(attrs)
        return cls.squeeze(), attrs


if __name__ == "__main__":
    # Used for initial testing of model.
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset, _ = load_unit_test()

    n_segments = 6
    alphabet_size = 4

    sax_aeon = SAXTransformer(n_segments=n_segments, alphabet_size=alphabet_size)

    x, _, _ = sax_aeon.transform(dataset)
    net = NeSyConceptLearner(n_input_dim = n_segments)

    output = net(x)
    print(output)



