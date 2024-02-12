import torch
from binaryLoader import BinaryDataset, pad_collate_func


def load_dataset(batch_size, max_seq_len=256, shuffle=True, num_workers=0):
    train_set = BinaryDataset(good_dir='/data1/ember2018/train/benign/',
                              bad_dir='/data1/ember2018/train/malware/',
                              max_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               collate_fn=pad_collate_func,
                                               drop_last=True)

    test_set = BinaryDataset(good_dir='/data1/ember2018/test/benign/',
                             bad_dir='/data1/ember2018/test/malware/',
                             max_len=max_seq_len)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=pad_collate_func,
                                              drop_last=True)

    return train_loader, test_loader
