from unittest import TestCase

import numpy as np
import scipy.signal
import tensorflow as tf

import filters


def naive_sepia(x):
    x = tf.cast(x, tf.float32)
    r, g, b = tf.split(x, 3, axis=-1)
    y = tf.concat((0.393 * r + 0.769 * g + 0.189 * b,
                   0.349 * r + 0.686 * g + 0.168 * b,
                   0.272 * r + 0.534 * g + 0.131 * b),
                  axis=-1)
    return y


def naive_grayscale(x):
    x = tf.cast(x, tf.float32)
    r, g, b = tf.split(x, 3, axis=-1)
    y = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return y[..., 0]


class FiltersTest(TestCase):
    def test_sepia(self):
        x = tf.random.normal((1, 10, 10, 3)) * 40 + 127.5
        x = tf.clip_by_value(x, 0, 255)

        expected = naive_sepia(x)
        actual = filters.sepia(x)

        np.testing.assert_array_equal(
            expected,
            actual,
        )

    def test_grayscale(self):
        x = tf.random.normal((1, 10, 10, 3)) * 40 + 127.5
        x = tf.clip_by_value(x, 0, 255)

        expected = naive_grayscale(x)
        actual = filters.grayscale(x)

        self.assertEqual(actual.shape, (1, 10, 10))

        np.testing.assert_array_equal(
            expected,
            actual
        )

    def test_correlation2d_3_x_3(self):
        s = np.arange(35).reshape(7, 5) + 1
        k = np.arange(9).reshape(3, 3)

        for mode, padding in (('same', 'SAME'), ('valid', 'VALID')):
            with self.subTest(f'padding {padding}'):
                np.testing.assert_array_equal(
                    np.expand_dims(scipy.signal.correlate2d(s, k, mode=mode), (0, -1)),
                    filters.correlate2d(s[np.newaxis, ...], k[..., np.newaxis], padding=padding)
                )

    def test_correlation2d_5_x_5(self):
        s = np.arange(35).reshape(7, 5) + 1
        k = np.arange(25).reshape(5, 5)

        for mode, padding in (('same', 'SAME'), ('valid', 'VALID')):
            np.testing.assert_array_equal(
                np.expand_dims(scipy.signal.correlate2d(s, k, mode=mode), (0, -1)),
                filters.correlate2d(s[np.newaxis, ...], k[..., np.newaxis], padding=padding)
            )

    def test_convolve2d_3_x_3(self):
        s = np.arange(35).reshape(7, 5) + 1
        k = np.arange(9).reshape(3, 3)

        for mode, padding in (('same', 'SAME'), ('valid', 'VALID')):
            with self.subTest(f'padding {padding}'):
                np.testing.assert_array_equal(
                    np.expand_dims(scipy.signal.convolve2d(s, k, mode=mode), (0, -1)),
                    filters.convolve2d(s[np.newaxis, ...], k[..., np.newaxis], padding=padding)
                )

    def test_convolve2d_5_x_5(self):
        s = np.arange(35).reshape(7, 5) + 1
        k = np.arange(25).reshape(5, 5)

        for mode, padding in (('same', 'SAME'), ('valid', 'VALID')):
            with self.subTest(f'padding {padding}'):
                np.testing.assert_array_equal(
                    np.expand_dims(scipy.signal.convolve2d(s, k, mode=mode), (0, -1)),
                    filters.convolve2d(s[np.newaxis, ...], k[..., np.newaxis], padding=padding)
                )

    def test_conv_broadcast(self):
        s = np.arange(70).reshape((2, 7, 5)) + 1
        k = np.asarray(
            [
                [[0., 2., 1.],
                 [0., 1., 0.]],

                [[0., 0., 0.],
                 [0., 1., 2.]]
            ]
        ).transpose((1, 2, 0))

        for padding in ('valid', 'same'):
            y_mine = filters.correlate2d(s, k, padding=padding.upper())
            y_scipy = np.asarray(
                [
                    [scipy.signal.correlate2d(s[b], k[..., c], mode=padding.lower())
                     for c in range(k.shape[-1])]
                    for b in range(s.shape[0])
                ]
            ).transpose((0, 2, 3, 1))  # transpose to (B, H, W, C) format.

            np.testing.assert_array_equal(
                y_mine,
                y_scipy
            )
