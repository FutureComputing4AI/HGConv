from utils import *
import flax.linen as nn
from dataset import load_dataset
from flax.training import train_state
from HRR.with_flax import Binding, Unbinding

from flax import jax_utils
from flax.training.common_utils import shard, shard_prng_key


class Network(nn.Module):
    vocab_size: int
    embed_size: int
    max_seq_len: int
    n_layer: int
    output_size: int
    dropout_rate: float

    def setup(self):
        self.binding = Binding()
        self.unbinding = Unbinding()

        weights = []
        for i in range(self.n_layer):
            weights.append(self.param(f'weight_{i}', jax.random.normal, (1, 1, self.embed_size)))
        self.weights = weights

    def conv(self, x, k, w):
        fk = np.fft.fft(k, n=x.shape[1], axis=1)
        k = np.fft.ifft(fk, n=x.shape[1], axis=1, norm='ortho')
        y = self.binding(x, k, axis=1) + x * w
        return nn.gelu(y)

    @nn.compact
    def __call__(self, encoder_input, training: bool = False):
        encoder_input = encoder_input.astype('int32')  # (B, T)
        en_mask = np.where(encoder_input > 0, 1., 0.)[:, :, np.newaxis]

        # embedding
        x = nn.Embed(self.vocab_size, self.embed_size)(encoder_input) * en_mask

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)

        for i in range(self.n_layer):
            ef = self.param(f'ef_{i}', jax.random.normal, (self.embed_size,))
            cf = self.param(f'cf_{i}', jax.random.normal, (1, 32, self.embed_size))
            df = self.param(f'df_{i}', jax.random.normal, (self.embed_size,))
            skip = x
            x = self.binding(x, ef, axis=-1)
            x = self.conv(x, cf, self.weights[i])
            x = self.unbinding(x, df, axis=-1)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.embed_size * 2)(x)
            x = nn.glu(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = x + skip

        # output
        x = np.sum(x * en_mask, axis=1) / np.sum(en_mask, axis=1)

        output = nn.Dense(self.output_size)(x)
        output = nn.log_softmax(output, axis=-1)
        return output, x


def initialize_model(model, input_size, init_rngs):
    init_inputs = np.ones([1, input_size])
    variables = model.init(init_rngs, init_inputs)['params']
    return variables


def cross_entropy_loss_(true, pred):
    true = true.astype('float')
    true = optax.smooth_labels(true, alpha=0.1)
    cross_entropy = np.sum(- true * pred, axis=-1)
    cross_entropy = np.nan_to_num(cross_entropy)
    return np.mean(cross_entropy)


def predict(state, batch, rngs):
    y_true = batch[1]
    y_pred, x = state.apply_fn({'params': state.params}, batch[0], rngs=rngs, training=False)
    loss = cross_entropy_loss_(true=y_true, pred=y_pred)
    acc = accuracy(y_true=y_true, y_pred=y_pred)
    metrics = {'loss': loss, 'accuracy': acc}
    return metrics, x


def train(batch_size, max_seq_len, embed_size=256, lr=1e-2, epochs=10):
    batch_size = batch_size * jax.device_count()
    vocab_size = 256 + 1
    n_layer = 1
    output_size = 20
    dropout_rate = 0.1
    name = 'drebin_tar'
    print('batch size:', batch_size, 'max seq len:', max_seq_len)

    # load dataset
    train_loader, test_loader = load_dataset(batch_size=batch_size,
                                             max_seq_len=max_seq_len,
                                             file_type="tars/",
                                             num_workers=10)

    # build and initialize network
    network = Network(vocab_size=vocab_size,
                      embed_size=embed_size,
                      max_seq_len=max_seq_len,
                      n_layer=n_layer,
                      output_size=output_size,
                      dropout_rate=dropout_rate)

    p_key_next, p_key = jax.random.split(jax.random.PRNGKey(0))
    d_key_next, d_key = jax.random.split(jax.random.PRNGKey(0))
    init_rngs = {'params': p_key, 'dropout': d_key}

    params = initialize_model(model=network, input_size=max_seq_len, init_rngs=init_rngs)

    # optimizer and scheduler
    steps = 8690 // batch_size
    scheduler = cosine_scheduler(base_lr=lr,
                                 steps=steps,
                                 warmup_epochs=1,
                                 epochs=epochs)

    tx = optax.adam(learning_rate=scheduler)
    state = train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    state = load_model(state, f'../weights/{name}_multi_{n_layer}_{max_seq_len}.h5')
    state = jax_utils.replicate(state)
    test_acc = []
    xs, ys = [], []

    for x_test, y_test in test_loader:
        p_key_next, p_key = jax.random.split(p_key_next)
        d_key_next, d_key = jax.random.split(d_key_next)
        rngs = {'params': shard_prng_key(p_key), 'dropout': shard_prng_key(d_key)}

        test_batch = [x_test, y_test]
        test_batch = shard(test_batch)

        metrics, x = jax.pmap(predict, axis_name="batch")(state, test_batch, rngs)
        test_acc.append(np.mean(metrics["accuracy"]))
        x = x.reshape((-1, embed_size))
        y = np.argmax(y_test, axis=-1)
        xs.append(x)
        ys.append(y)

    print(f"Accuracy: {sum(test_acc) / len(test_acc)}")
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)

    np.save(f"../data/{name}_x.npy", xs)
    np.save(f"../data/{name}_y.npy", ys)


if __name__ == '__main__':
    import os

    os.environ["TF_USE_NVLINK_FOR_PARALLEL_COMPILATION"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(batch_size=32, max_seq_len=4096, epochs=20)
