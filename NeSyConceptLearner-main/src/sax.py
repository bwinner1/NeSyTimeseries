import datetime
import matplotlib.pyplot as plt
import numpy as np
from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_unit_test
from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler
import datasets


#################
### SAX       ###
#################

class SAXTransformer:
    def __init__(self, n_segments=8, alphabet_size=4):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.sax = SAX(self.n_segments, self.alphabet_size)

    def transform(self, dataset):
        # Make sure that number of time steps is dividable by n_segments
        remainder = dataset.shape[2] % self.n_segments
        if(remainder != 0):
            fillers = self.n_segments - remainder
            meansToFill = np.repeat(np.mean(dataset, axis=2), fillers, axis=1)
            dataset = np.append(dataset[:,0,:], meansToFill, axis = 1) 
            # append function automatically removes all 1s in array shape, add it manually:  
            dataset = np.expand_dims(dataset, 1)
            print(f"Added {self.getEntity(fillers, 'mean')}")
            
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Fit the scaler on the training data (learn the mean and standard deviation)
        # Transform both the training and test data
        # Transposition on input and output added, as StandardScaler operates column-wise (e.g. calculates mean of first column)
        dataset_scaled = scaler.fit_transform(dataset[:, 0, :].T).T

        # Fit and transform the data
        dataset_sax = self.sax.fit_transform(dataset_scaled)
        
        return dataset, dataset_sax, dataset_scaled

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
    def drawGrid(self, dataset, dataset_scaled, dataset_sax, rows=2, columns=3, dataset_name="example_dataset"):

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
        






#################
### Testing Dataset
#################

dataset = np.array([[[0, 1, 2, 3, 6]],
                    [[0, 1, 2, 3, 6]],
                    [[0, 1, 2, 3, 6]],
                    [[0, 6, 3, 2, 1]],
                    [[0, 1, 2, 3, 6]],
                    [[0, 1, 2, 3, 6]],
                    [[0, 1, 2, 3, 6]],
                    [[0, 6, 3, 2, 1]],])

sax_test = SAXTransformer(n_segments=3, alphabet_size=6)
dataset, dataset_sax, dataset_scaled = sax_test.transform(dataset)
sax_test.drawGrid(dataset, dataset_scaled, dataset_sax, rows=2, columns=4, dataset_name="testdataset")

print("Testing dataset done")

#################
### Aeon Initial Dataset
#################

dataset, _ = load_unit_test()

sax_aeon = SAXTransformer(n_segments=6, alphabet_size=8)
dataset, dataset_sax, dataset_scaled = sax_aeon.transform(dataset)
# Alternative:
# sax_aeon.printGeneralInfo(dataset, dataset_scaled, dataset_sax, increment, time_steps)
# dataset_sax = sax_aeon.transform(dataset)

sax_aeon.drawGrid(dataset, dataset_scaled, dataset_sax, dataset_name="unit_test") #rows=2, columns=4



#################
### Aeon Wine Dataset
#################
from aeon.datasets import load_classification

dataset, y, meta_data = load_classification("Wine", return_metadata=True)

sax_wine = SAXTransformer(n_segments=6, alphabet_size=8)
dataset, dataset_sax, dataset_scaled = sax_wine.transform(dataset)
sax_wine.drawGrid(dataset, dataset_scaled, dataset_sax, dataset_name="wine") #rows=2, columns=4



#################
### P2S Dataset
#################

from datasets import load_dataset
dataset = load_dataset('AIML-TUDA/P2S', 'Decoy', download_mode='reuse_dataset_if_exists')

# Print the first example in the dataset
# print(dataset['train'][0])  # Use 'train', 'test', etc. as needed

X = dataset['train']

print(X.shape)
# (1154, 4)
#print(X[0])
