import numpy
import jax.numpy as np
import flax.linen as nn


class EmbeddingLearned(nn.Module):
    vocab_size: int
    embed_size: int
    max_seq_len: int

    @nn.compact
    def __call__(self, inputs):
        length = inputs.shape[-1]
        positions = np.arange(start=0, stop=length, step=1)
        word_embedding = nn.Embed(self.vocab_size, self.embed_size)(inputs)
        position_embedding = nn.Embed(self.max_seq_len, self.embed_size)(positions)
        return word_embedding + position_embedding


class EmbeddingFixed(nn.Module):
    vocab_size: int
    embed_size: int
    max_seq_len: int

    def setup(self, min_scale=1.0, max_scale=10000.0):
        pe = numpy.zeros((self.max_seq_len, self.embed_size), dtype=numpy.float32)
        position = numpy.arange(0, self.max_seq_len)[:, numpy.newaxis]
        scale_factor = -numpy.log(max_scale / min_scale) / (self.embed_size // 2 - 1)
        div_term = min_scale * numpy.exp(numpy.arange(0, self.embed_size // 2) * scale_factor)
        pe[:, :self.embed_size // 2] = numpy.sin(position * div_term)
        pe[:, self.embed_size // 2: 2 * (self.embed_size // 2)] = numpy.cos(position * div_term)
        pe = pe[numpy.newaxis, :, :]
        self.pe = np.array(pe)
        self.we = nn.Embed(self.vocab_size, self.embed_size)

    @nn.compact
    def __call__(self, x):
        word_embedding = self.we(x)
        positional_embedding = self.pe[:, 0:x.shape[1], :]
        return word_embedding + positional_embedding

    def attend(self, y):
        return self.we.attend(y)


class PositionalEmbedding(nn.Module):
    embed_size: int
    max_seq_len: int

    def setup(self, min_scale=1.0, max_scale=10000.0):
        pe = numpy.zeros((self.max_seq_len, self.embed_size), dtype=numpy.float32)
        position = numpy.arange(0, self.max_seq_len)[:, numpy.newaxis]
        scale_factor = -numpy.log(max_scale / min_scale) / (self.embed_size // 2 - 1)
        div_term = min_scale * numpy.exp(numpy.arange(0, self.embed_size // 2) * scale_factor)
        pe[:, :self.embed_size // 2] = numpy.sin(position * div_term)
        pe[:, self.embed_size // 2: 2 * (self.embed_size // 2)] = numpy.cos(position * div_term)
        pe = pe[numpy.newaxis, :, :]
        self.pe = np.array(pe)

    @nn.compact
    def __call__(self, x):
        positional_embedding = self.pe[:, 0:x.shape[1], :]
        return x + positional_embedding
