import csv
import jax
import numpy
import optax
import pickle
import pandas as pd
import jax.numpy as np
from flax import serialization
from flax.core.frozen_dict import freeze, unfreeze


def split(x, heads):
    b, t, h = x.shape
    x = x.reshape(b, t, heads, h // heads)
    return x.transpose((0, 2, 1, 3))


def merge(x):
    b, heads, t, h = x.shape
    x = x.transpose((0, 2, 1, 3))
    return x.reshape(b, t, heads * h)


def positional_embedding(key, d, dtype=None):
    max_seq_len, embed_size = d
    pe = numpy.zeros((max_seq_len, embed_size), dtype=numpy.float32)
    position = numpy.arange(0, max_seq_len)[:, numpy.newaxis]
    div_term = numpy.exp(numpy.arange(0, embed_size, 2) * -(numpy.log(10000.0) / embed_size))
    pe[:, 0::2] = numpy.sin(position * div_term)
    pe[:, 1::2] = numpy.cos(position * div_term)
    # pe = pe[numpy.newaxis, :, :]  # [1, T, H]
    return np.array(pe)


def cosine_scheduler(base_lr, steps, warmup_epochs, epochs):
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_lr,
        transition_steps=warmup_epochs * steps)
    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=cosine_epochs * steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps])
    return schedule_fn


def look_ahead_mask(x):
    ones = np.ones(shape=(x.shape[1], x.shape[1]))
    mask = np.expand_dims(np.tril(ones), axis=0)
    return np.repeat(mask, x.shape[0], axis=0)


def one_hot(x, n_class):
    return (np.arange(n_class) == x[..., None]).astype(int)


def cross_entropy_loss(y_true, y_pred, eps=1e-10):
    y_t = np.clip(y_true, eps, 1.)
    y_p = np.clip(y_pred, eps, 1.)
    y_t_prime = np.clip(1. - y_true, eps, 1.)
    y_p_prime = np.clip(1. - y_pred, eps, 1.)
    bce = - y_t * np.log(y_p) - y_t_prime * np.log(y_p_prime)
    loss = np.sum(bce, axis=-1)
    return np.mean(loss)


def evaluate(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    accuracies = np.equal(y_true, y_pred)
    mask = np.logical_not(np.equal(y_true, 0)).astype(np.float32)
    accuracies = np.logical_and(mask, accuracies).astype(np.float32)
    return np.sum(accuracies) / np.sum(mask) * 100.


def accuracy(y_true, y_pred):
    acc = np.argmax(y_true, -1) == np.argmax(y_pred, -1)
    return np.mean(acc) * 100


def l2_regularization(params, alpha=1.0):
    x2 = jax.tree_map(lambda x: np.square(x).mean(), params)
    loss = np.asarray(jax.tree_leaves(x2)).sum()
    return alpha * loss


def grad_check(grads):
    grads = unfreeze(grads)
    grads = jax.tree_map(lambda x: np.nan_to_num(x), grads)
    return freeze(grads)


def index_sequence(batch_size, dataset_size):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    return list(zip(index_a, index_b))


def bias_initializer(key, shape, dtype=np.float32):
    if key is not None:
        pass
    return np.zeros(shape, dtype)


def load_model(state, name):
    with open(name, 'rb') as f:
        dict_state = pickle.loads(f.read())
    return serialization.from_state_dict(state, dict_state)


def save_model(state, name):
    dict_state = serialization.to_state_dict(state)
    with open(name, 'wb') as f:
        pickle.dump(dict_state, f)


def save_history(file, history, mode='w'):
    with open(file, mode) as f:
        writer = csv.writer(f)
        history = [line.replace(':', ',').split(',') for line in history]
        [writer.writerow(line) for line in history]


def save_history_to_csv(file, tr_acc, tr_loss, te_acc, te_loss):
    columns = ['train acc', 'train loss', 'test acc', 'test loss']
    results = pd.DataFrame(data=list(zip(tr_acc, tr_loss, te_acc, te_loss)), columns=columns)
    results.to_csv(file or None, index=False)
