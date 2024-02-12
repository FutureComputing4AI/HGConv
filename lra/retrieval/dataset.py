import tensorflow_datasets as tfds
from lra_benchmarks.matching.input_pipeline import get_matching_datasets


def load_dataset(batch_size, max_seq_len, n_devices=1):
    data_dir = './../lra_release/lra_release/tsv_data'
    train_dataset, val_dataset, test_dataset, encoder = get_matching_datasets(n_devices=n_devices,
                                                                              task_name=None,
                                                                              data_dir=data_dir,
                                                                              batch_size=batch_size,
                                                                              fixed_vocab=None,
                                                                              max_length=max_seq_len,
                                                                              tokenizer='char',
                                                                              vocab_file_path=None)

    train_dataset = tfds.as_numpy(train_dataset)
    val_dataset = tfds.as_numpy(val_dataset)
    test_dataset = tfds.as_numpy(test_dataset)
    return train_dataset, val_dataset, test_dataset
