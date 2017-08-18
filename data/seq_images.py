import h5py
import numpy as np
from fuel.datasets import H5PYDataset


def seq_images(path, source, file_names, target_sequence, num_examples=None,
               seed=None):
    """TODO: Docstring for seq_images.

    :arg1: TODO
    :returns: TODO

    """
    if not seed:
        seed = 123

    rng = np.random.RandomState(seed)

    with h5py.File(name=source, mode='r') as f:
        targets = f.get(file_names[-1]).value
        features = f.get(file_names[0]).value

        if not num_examples:
            num_examples = f.get(file_names[-1]).shape[0]

        labels = set(targets.flatten().tolist())

        labels_pool = {}

        seq_index = []
        new_features = []
        for i in labels:
            labels_pool[i] = (np.arange(targets.shape[0]).reshape(
                targets.shape[0], 1)[targets == i]).tolist()

        for i in target_sequence:
            seq_index.append(rng.choice(labels_pool[i], size=num_examples))
            new_features.append(features[seq_index[-1]])

        new_features = np.stack(new_features, 1)
        print new_features.shape

        with h5py.File(path, 'w') as g:
            data_buffer = new_features.astype('float32')
            data_targets = np.array(target_sequence).astype('int32')

            features = g.create_dataset('features', (num_examples,
                                                     len(target_sequence), 1,
                                                     28, 28),
                                        dtype='float32')
            targets = g.create_dataset('targets',
                                       (len(target_sequence),), dtype='int32')
# assign the data
            features[...] = data_buffer
            targets[...] = data_targets

            split_dict = {
                'train': {'features': (0, 60000),
                          'targets': (0, 0)},
                'valid': {'features': (60000, 70000),
                          'targets': (0, 0)},
                'test': {'features': (70000, num_examples),
                         'targets': (0, 0)}
                        }
            g.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            g.flush()

if __name__ == "__main__":
    seq_images(path='./seq_mnist.hdf5', source='./data/mnist.hdf5',
               file_names=('features', 'targets'), target_sequence=range(10),
               num_examples=None,
               seed=123)
