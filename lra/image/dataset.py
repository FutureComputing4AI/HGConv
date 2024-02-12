import tensorflow_datasets as tfds
from lra_benchmarks.image.input_pipeline import get_cifar10_datasets


def load_dataset(batch_size, n_devices=1):
    train_loader, valid_loader, test_loader, _, _, _ = get_cifar10_datasets(n_devices=n_devices,
                                                                            batch_size=batch_size,
                                                                            normalize=False)
    train_loader = tfds.as_numpy(train_loader)
    valid_loader = tfds.as_numpy(valid_loader)
    test_loader = tfds.as_numpy(test_loader)

    return train_loader, valid_loader, test_loader
