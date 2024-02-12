import os
import torch
import random
import numpy as np
from torch.nn.functional import one_hot


class DrebinDataset(torch.utils.data.Dataset):
    def __init__(self, split, seed, file_type, max_len=128_000):
        self.directory = "/data3/drebin/"
        self.file_type = file_type
        classes = os.listdir(self.directory + file_type)

        meta = {}
        for c in classes:
            meta[c] = len(os.listdir(self.directory + file_type + c))

        top_20_classes = list(sorted(meta.items(), key=lambda x: x[1], reverse=True)[0:20])

        lines = []

        for i, (name, item) in enumerate(top_20_classes):
            files = []
            for file in os.listdir(self.directory + file_type + name):
                files.append([f"{name}/{file}", i])
            lines += files

        x, y = {}, {}
        for line in lines:
            h, c = line
            if str(c) in x:
                x[str(c)].append(h)
                y[str(c)].append(c)
            else:
                x[str(c)] = [h]
                y[str(c)] = [c]

        self.x_true = []
        self.y_true = []
        for i in range(20):
            n = int(len(y[str(i)]) * 0.8)
            if split == "train":
                self.x_true += x[str(i)][:n]
                self.y_true += y[str(i)][:n]
            elif split == "test":
                self.x_true += x[str(i)][n:]
                self.y_true += y[str(i)][n:]
            else:
                raise NameError

        self.max_len = max_len

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, index):
        with open(self.directory + self.file_type + self.x_true[index], mode="rb") as f:
            x = f.read(self.max_len)
            x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1

        return torch.tensor(x), torch.tensor([self.y_true[index]])


def pad_collate_func(batch):
    true = [x[0] for x in batch]
    pred = [x[1] for x in batch]
    x = torch.nn.utils.rnn.pad_sequence(true, batch_first=True)
    y = torch.stack(pred)[:, 0]
    y = one_hot(y, num_classes=20)
    return x.numpy(), y.numpy()


def load_dataset(batch_size, max_seq_len=256, file_type="apks/", num_workers=0, drop_last=False):
    if file_type not in ["apks/", "tars/"]:
        raise NameError("file type not allowed. available file types are apks/ or tars/.")
    seed = random.randint(0, 1000)
    train_set = DrebinDataset(split="train", seed=seed, file_type=file_type, max_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=pad_collate_func,
                                               drop_last=drop_last)

    test_set = DrebinDataset(split="test", seed=seed, file_type=file_type, max_len=max_seq_len)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=pad_collate_func,
                                              drop_last=drop_last)

    return train_loader, test_loader
