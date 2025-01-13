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

from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler

class SlotAttention(nn.Module):
    """
    Implementation from https://github.com/lucidrains/slot-attention by lucidrains.
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim)).abs().to(device='cuda')

        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim, eps=1e-05)
        self.norm_slots = nn.LayerNorm(dim, eps=1e-05)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-05)

        # dummy initialisation
        self.attn = 0

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_inputs(inputs)
        k, v = self.project_k(inputs), self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        self.attn = attn

        return slots


class SlotAttention_encoder(nn.Module):
    """
    Slot attention encoder for CLEVR as in Locatello et al. 2020 according to the set prediction architecture.
    """
    def __init__(self, in_channels, hidden_channels):
        """
        Builds the Encoder for the set prediction architecture
        :param in_channels: Integer, input channel dimensions to encoder
        :param hidden_channels: Integer, hidden channel dimensions within encoder
        """
        super(SlotAttention_encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    """
    MLP for CLEVR as in Locatello et al. 2020 according to the set prediction architecture.
    """
    def __init__(self, hidden_channels):
        """
        Builds the MLP
        :param hidden_channels: Integer, hidden channel dimensions within encoder, is also equivalent to the input
        channel dims here.
        """
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.network(x)


def build_grid(resolution):
    """
    Builds the grid for the Posisition Embedding.
    :param resolution: Tuple of Ints, in the dimensions of the latent space of the encoder.
    :return: 2D Float meshgrid representing th x y position.
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):
    """
    Adds soft positional embedding with learnable projection.
    """
    def __init__(self, hidden_size, resolution, device="cuda"):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
          device: String specifiying the device, cpu or cuda
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.grid = torch.FloatTensor(build_grid(resolution))
        self.grid = self.grid.to(device)
        self.resolution = resolution[0]
        self.hidden_size = hidden_size

    def forward(self, inputs):
        return inputs + self.dense(self.grid).view((-1, self.hidden_size, self.resolution, self.resolution))


class SlotAttention_classifier(nn.Module):
    """
    The classifier of the set prediction architecture of Locatello et al. 2020
    """
    def __init__(self, in_channels, out_channels):
        """
        Builds the classifier for the set prediction architecture.
        :param in_channels: Integer, input channel dimensions
        :param out_channels: Integer, output channel dimensions
        """
        super(SlotAttention_classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class SlotAttention_model(nn.Module):
    """
    The set prediction slot attention architecture for CLEVR as in Locatello et al 2020.
    """
    def __init__(self, n_slots, n_iters, n_attr, category_ids,
                 in_channels=3,
                 encoder_hidden_channels=64,
                 attention_hidden_channels=128,
                 device="cuda"):
        """
        Builds the set prediction slot attention architecture.
        :param n_slots: Integer, number of slots
        :param n_iters: Integer, number of attention iterations
        :param n_attr: Integer, number of attributes per object to predict
        :param category_ids: List of Integers, specifying the boundaries of each attribute group, e.g. color
        attributes are variables 10 to 17
        :param in_channels: Integer, number of input channel dimensions
        :param encoder_hidden_channels: Integer, number of encoder hidden channel dimensions
        :param attention_hidden_channels: Integer, number of hidden channel dimensions for slot attention module
        :param device: String, either 'cpu' or 'cuda'
        """
        super(SlotAttention_model, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.n_attr = n_attr
        self.category_ids = category_ids
        self.n_attr = n_attr + 1  # additional slot to indicate if it is a object or empty slot
        self.device = device

        self.encoder_cnn = SlotAttention_encoder(in_channels=in_channels, hidden_channels=encoder_hidden_channels)
        self.encoder_pos = SoftPositionEmbed(encoder_hidden_channels, (32, 32), device=device)
        self.layer_norm = nn.LayerNorm(encoder_hidden_channels, eps=1e-05)
        self.mlp = MLP(hidden_channels=encoder_hidden_channels)
        self.slot_attention = SlotAttention(num_slots=n_slots, dim=encoder_hidden_channels, iters=n_iters, eps=1e-8,
                                            hidden_dim=attention_hidden_channels)
        self.mlp_classifier = SlotAttention_classifier(in_channels=encoder_hidden_channels, out_channels=self.n_attr)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_pos(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = self.slot_attention(x)
        x = self.mlp_classifier(x)
        return x

    def _transform_attrs(self, attrs):
        """
        receives the attribute predictions and binarizes them by computing the argmax per attribute group
        :param attrs: 3D Tensor, [batch, n_slots, n_attrs] attribute predictions for a batch.
        :return: binarized attribute predictions
        """
        presence = attrs[:, :, 0]
        attrs_trans = attrs[:, :, 1:]

        # threshold presence prediction, i.e. where is an object predicted
        presence = presence < 0.5

        # flatten first two dims
        attrs_trans = attrs_trans.view(1, -1, attrs_trans.shape[2]).squeeze()
        # binarize attributes
        # set argmax per attr to 1, all other to 0, s.t. only zeros and ones are contained within graph
        # NOTE: this way it is not differentiable!
        bin_attrs = torch.zeros(attrs_trans.shape, device=self.device)
        for i in range(len(self.category_ids) - 1):
            # find the argmax within each category and set this to one
            bin_attrs[range(bin_attrs.shape[0]),
                      # e.g. x[:, 0:(3+0)], x[:, 3:(5+3)], etc
                      (attrs_trans[:,
                       self.category_ids[i]:self.category_ids[i + 1]].argmax(dim=1) + self.category_ids[i]).type(
                          torch.LongTensor)] = 1

        # reshape back to batch x n_slots x n_attrs
        bin_attrs = bin_attrs.view(attrs.shape[0], attrs.shape[1], attrs.shape[2] - 1)

        # add coordinates back
        bin_attrs[:, :, :3] = attrs[:, :, 1:4]

        # redo presence zeroing
        bin_attrs[presence, :] = 0

        return bin_attrs


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

        #print("self.sax.get_fitted_params")
        #print(self.sax.get_fitted_params())

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
    """ 
    def __init__(self, n_segments=8, alphabet_size=4):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.sax = SAX(self.n_segments, self.alphabet_size)
 """
    @staticmethod
    def transform(dataset, y, filtered_columns=None, setting="min"):
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
                    "middle" : EfficientFCParameters(),
                    "fast" : MinimalFCParameters()}
        # Only extract
        X = extract_features(df, column_id='ts_id', impute_function=impute,
                              default_fc_parameters=fcparams[setting])
        
        """
        # Extract and filter
        # X = extract_relevant_features(df, y, column_id='ts_id',
        #                                default_fc_parameters=MinimalFCParameters())
         """
        
        #If no columns are given (train dataset), then use select_features from tsfresh package
        if filtered_columns is None:
            X_filtered = select_features(X, y)
            filtered_columns = X_filtered.columns
        
        # If columns are given (train, val), then select the same for the train and test dataset
        else:
            X_filtered = X[filtered_columns]
        # print(X.shape)

        # Convert the pd to a tensor, add an inner dimension
        return torch.tensor(X_filtered.values, dtype=torch.float32).unsqueeze(1), filtered_columns



############
# NeSyConceptLearner #
############
class NeSyConceptLearner(nn.Module):
    """
    The Neuro-Symbolic Concept Learner of Stammer et al. 2021 based on Slot Attention and Set Transformer.
    """
    def __init__(self, n_attr, n_classes = 2, n_set_heads = 4, set_transf_hidden = 128,
                 device='cuda'):
        """ 
        For BCEWithLogitsLoss n_classes is 1, as network should have a binary output, either defect or not
        Note: set_transf_hidden % n_set_heads == 0 has to true!

        #old version
    def __init__(self, n_classes, n_slots=1, n_iters=3, n_attr=18, n_set_heads=4, set_transf_hidden=128,
                 category_ids=[3, 6, 8, 10, 17], device='cuda'): """
        """
        :param n_classes: Integer, number of classes
        :param n_attr: Integer, number of attributes per object
        :param n_set_heads: Integer, number of attention heads for set transformer
        :param set_transf_hidden: Integer, hidden dim of set transformer
        prediction
        
        :param device: String, either 'cpu' or 'cuda'
        """
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_attr = n_attr
        
        # Concept Embedding Module
        # --- left out; is applied before the network ---

        # Reasoning module
        self.set_cls = SetTransformer(dim_input=n_attr, dim_hidden=set_transf_hidden, num_heads=n_set_heads,
                                      dim_output=n_classes, ln=True)

    def forward(self, attrs):
        #TODO: modify method comment
        """
        Receives an image, passes it through the concept embedding module and the reasoning module. For simplicity we
        here binarize the continuous output of the concept embedding module before passing it to the reasoning module.
        The ouputs of both modules are returned, i.e. the final class prediction and the latent symbolic representation.
        :param img: 4D Tensor [batch, channels, width, height]
        :return: Tuple of outputs of both modules, [batch, n_classes] classification/ reasoning module output,
        [batch, n_slots, n_attr] concept embedding module output/ symbolic representation
        """
        """
        :param attrs: 3D Tensor[batch, sets, features]
        """
        attrs = attrs.float()
        cls = self.set_cls(attrs)
        return cls.squeeze(), attrs


if __name__ == "__main__":
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #x = torch.rand(20, 3, 128, 128).to(device)
    dataset, _ = load_unit_test()

    n_segments = 6
    alphabet_size = 4

    sax_aeon = SAXTransformer(n_segments=n_segments, alphabet_size=alphabet_size)
    #dataset_sax, dataset, dataset_scaled = sax_aeon.transform(dataset)
    #sax_aeon.drawGrid(dataset, dataset_scaled, dataset_sax, dataset_name="unit_test") #rows=2, columns=4

    x, _, _ = sax_aeon.transform(dataset)
    
    #n_classes is 2, as there are two possible outcomes for a sample, either defect or not 

    net = NeSyConceptLearner(n_attr = n_segments)
    #net = NeSyConceptLearner(n_classes=2, n_attr=6, n_set_heads=4, set_transf_hidden=128,
    #                         device=device).to(device)
    output = net(x)
    print(output)



