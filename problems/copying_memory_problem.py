import numpy as np
from .dataset import Dataset
import tensorflow.logging as log

# single output = False
# num_in = 1
# num_target = 1

class CopyingMemoryProblemDataset(Dataset):
    def generate(self, num_samples, seed=True):

        assert(self.sample_len > 20) # must be

        # for reproducibility
        if seed:
            log.info("Seeding data samples for copying memory problem...")
            np.random.seed(42)

        X = np.zeros((num_samples, self.sample_len, 1))
        data = np.random.randint(low = 1, high = 9, size = (num_samples, 10, 1))
        X[:, :10] = data
        X[:, -11] = 9
        Y = np.zeros((num_samples, self.sample_len, 1))
        Y[:, -10:] = X[:, :10]
        return X, Y
