import tensorflow_datasets as tfds
from lra_benchmarks.text_classification.input_pipeline import get_tc_datasets


def load_dataset(batch_size, max_seq_len, n_devices=1):
    train_dataset, val_dataset, test_dataset, encoder = get_tc_datasets(n_devices=n_devices,
                                                                        task_name='imdb_reviews',
                                                                        data_dir=None,
                                                                        batch_size=batch_size,
                                                                        fixed_vocab=None,
                                                                        max_length=max_seq_len,
                                                                        tokenizer='char')

    train_dataset = tfds.as_numpy(train_dataset)
    val_dataset = tfds.as_numpy(val_dataset)
    test_dataset = tfds.as_numpy(test_dataset)
    return train_dataset, val_dataset, test_dataset
