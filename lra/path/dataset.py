import tensorflow_datasets as tfds
from lra_benchmarks.image.input_pipeline import get_pathfinder_base_datasets


def load_dataset(batch_size, resolution, split, n_devices=1):
    train_loader, valid_loader, test_loader, _, _, inputs_shape = get_pathfinder_base_datasets(n_devices=n_devices,
                                                                                               batch_size=batch_size,
                                                                                               resolution=resolution,
                                                                                               normalize=False,
                                                                                               split=split,
                                                                                               _PATHFINER_TFDS_PATH='./../lra_release/TFDS/')
    train_loader = tfds.as_numpy(train_loader)
    valid_loader = tfds.as_numpy(valid_loader)
    test_loader = tfds.as_numpy(test_loader)
    return train_loader, valid_loader, test_loader
