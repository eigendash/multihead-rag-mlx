import mlx.core as mx
import numpy as np

from mrag.model import Encoder, EncoderConfig


def make(seed=0):
    mx.random.seed(seed)
    model = Encoder(EncoderConfig(vocab=50, dim=32, heads=4, layers=3, max_len=32))
    mx.eval(model.parameters())
    return model


def test_encode_shapes():
    model = make()
    tokens = mx.random.randint(2, 50, (5, 12))
    final, heads = model.encode(tokens, mx.array([12, 12, 12, 12, 12]))
    assert final.shape == (5, 32) and heads.shape == (5, 4, 8)


def test_heads_are_the_inputs_of_the_output_projection():
    # The head vectors must be what the last block's output projection merges.
    model = make()
    tokens = mx.random.randint(2, 50, (3, 10))
    x = model.tok(tokens) + model.pos(mx.arange(10))
    for block in model.blocks[:-1]:
        x, _ = block(x)
    last = model.blocks[-1]
    attn_out, per_head = last.attn(last.norm1(x))
    t = 6
    merged = per_head[:, :, t].reshape(3, -1)
    np.testing.assert_allclose(np.array(last.attn.out(merged)), np.array(attn_out[:, t]), atol=1e-5)


def test_variable_length_rows_read_their_own_last_token():
    model = make()
    a = mx.random.randint(2, 50, (1, 7))
    b = mx.random.randint(2, 50, (1, 12))
    padded = mx.concatenate([mx.pad(a, [(0, 0), (0, 5)]), b], axis=0)
    final, heads = model.encode(padded, mx.array([7, 12]))
    final_a, heads_a = model.encode(a, mx.array([7]))
    np.testing.assert_allclose(np.array(final[0]), np.array(final_a[0]), atol=1e-5)
    np.testing.assert_allclose(np.array(heads[0]), np.array(heads_a[0]), atol=1e-5)


def test_embedding_depends_on_earlier_tokens():
    model = make()
    a = mx.array([[5, 6, 7, 8, 1]])
    b = mx.array([[5, 9, 7, 8, 1]])
    fa, ha = model.encode(a, mx.array([5]))
    fb, hb = model.encode(b, mx.array([5]))
    assert not np.allclose(np.array(fa), np.array(fb), atol=1e-4)
    assert not np.allclose(np.array(ha), np.array(hb), atol=1e-4)
