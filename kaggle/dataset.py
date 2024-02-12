import csv
import torch
import random
import numpy as np
from torch.nn.functional import one_hot


class MicrosoftMalwareDataset(torch.utils.data.Dataset):
    def __init__(self, split, seed=0, raw=False, max_len=128_000):
        self.directory = "../../../microsoft_malware_classification/"
        with open(self.directory + "trainLabels.csv", mode="r") as f:
            reader = csv.reader(f)
            lines = [row for row in reader][1:]
            random.seed(seed)
            random.shuffle(lines)

        x, y = {}, {}
        for line in lines:
            h, c = line
            if c in x:
                x[c].append(h)
                y[c].append(int(c) - 1)
            else:
                x[c] = [h]
                y[c] = [int(c) - 1]

        self.x_true = []
        self.y_true = []
        for i in range(1, 10):
            n = int(len(y[str(i)]) * 0.8)
            if split == "train":
                self.x_true += x[str(i)][:n]
                self.y_true += y[str(i)][:n]
            elif split == "test":
                self.x_true += x[str(i)][n:]
                self.y_true += y[str(i)][n:]
            else:
                raise NameError

        if raw:
            self.extension = ".bytes"
        else:
            self.extension = ".asm"

        self.max_len = max_len

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, index):
        with open(self.directory + f"train/{self.x_true[index]}{self.extension}", mode="rb") as f:
            x = f.read(self.max_len)
            x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1

        return torch.tensor(x), torch.tensor([self.y_true[index]])


def pad_collate_func(batch):
    true = [x[0] for x in batch]
    pred = [x[1] for x in batch]
    x = torch.nn.utils.rnn.pad_sequence(true, batch_first=True)
    y = torch.stack(pred)[:, 0]
    y = one_hot(y, num_classes=9)
    return x.numpy(), y.numpy()


def load_dataset(batch_size, max_seq_len=256, raw=True, num_workers=0):
    seed = random.randint(0, 1000)
    train_set = MicrosoftMalwareDataset(split="train", seed=seed, raw=raw, max_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=pad_collate_func,
                                               drop_last=True)

    test_set = MicrosoftMalwareDataset(split="test", seed=seed, raw=raw, max_len=max_seq_len)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=pad_collate_func,
                                              drop_last=True)

    return train_loader, test_loader
