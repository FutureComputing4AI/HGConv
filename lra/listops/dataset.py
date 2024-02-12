import tensorflow_datasets as tfds
from lra_benchmarks.listops.input_pipeline import get_datasets


def load_dataset(batch_size, n_devices=1):
    train_dataset, val_dataset, test_dataset, encoder = get_datasets(n_devices=n_devices,
                                                                     task_name='basic',
                                                                     data_dir='./../lra_release/listops-1000/',
                                                                     batch_size=batch_size,
                                                                     max_length=2000)

    train_dataset = tfds.as_numpy(train_dataset)
    valid_dataset = tfds.as_numpy(val_dataset)
    test_dataset = tfds.as_numpy(test_dataset)
    return train_dataset, valid_dataset, test_dataset
