"""Apply the Discrete Fourier Transform to images.

Author: Lucas David -- <lucas.david@ic.unicamp.br>

"""

import os
import pathlib
import sys
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from PIL import Image

tf.get_logger().setLevel('ERROR')

SPATIAL_AXES = (-1, -2)


# region filters

def circ_filter2d(radius, shape, dtype=tf.float32):
    h, w = shape[-2:]

    ih = tf.abs(tf.range(h) - h // 2)  # [... 2 1 0 1 2 ...]^T
    ih = tf.reshape(ih, (-1, 1))
    iw = tf.abs(tf.range(w) - w // 2)  # [... 2 1 0 1 2 ...]

    radius = as_absolute_length(radius, h, w)
    radius = tf.reshape(radius, (-1, 1, 1))

    return tf.cast(ih ** 2 + iw ** 2 < radius ** 2, dtype)


def compression_filter2d(rate, s):
    m = tf.abs(s)
    th = tfp.stats.percentile(m, rate, axis=SPATIAL_AXES)
    m = tf.cast(m > th, tf.complex64)

    return m


circ_filter2d_like = lambda radius, a: circ_filter2d(radius, a.shape, a.dtype)


# endregion


# region exercises


def transform(
        image,
        low_pass,
        high_pass,
        band_pass,
        compression_rate,
        rotation
):
    y = tf.cast(image, tf.complex64)
    y = tf.signal.fft2d(y[..., 0])
    y = tf.signal.fftshift(y, axes=SPATIAL_AXES)

    lf = circ_filter2d_like(low_pass, y)
    hf = 1 - circ_filter2d_like(high_pass, y)
    bf = circ_filter2d_like(max(band_pass), y) - circ_filter2d_like(min(band_pass), y)
    cf = compression_filter2d(compression_rate, y)

    rotated = tfa.image.rotate(image, np.pi*rotation/180.)
    r = tf.cast(rotated, tf.complex64)
    r = tf.signal.fft2d(r[..., 0])
    r = tf.signal.fftshift(r, axes=SPATIAL_AXES)

    return ((reconstruct(y), y),
            (reconstruct(lf * y), lf * y),
            (reconstruct(hf * y), hf * y),
            (reconstruct(bf * y), bf * y),
            (reconstruct(cf * y), cf * y),
            (rotated, r))


# endregion

# region Util functions


def as_absolute_length(measures, height, width):
    if isinstance(measures, (int, float)):
        measures = [measures]

    return tf.convert_to_tensor([
        (m
         if isinstance(m, int)
         else int(m * min(height, width) // 2))
        for m in measures
    ])


def load_images(
        path: str
) -> List[Tuple[str, tf.Tensor]]:
    """Load multiple images as a list of tensors.

    Arguments
    ---------
    path: str
      path to image file or directory

    Returns
    -------
    List of pairs (filename, image tensor).

    """
    print(f'Loading images from {path}...')

    if not os.path.exists(path):
        raise FileExistsError(path)

    if os.path.isfile(path):
        files = [path]
    else:
        files = list(map(str, pathlib.Path(path).glob('*')))

    images = [(f, load_image(f)) for f in files]
    print(f'{len(images)} found')

    return images


def load_image(
        path: str
) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def reconstruct(signal):
    z = tf.signal.ifftshift(signal, axes=SPATIAL_AXES)
    z = tf.signal.ifft2d(z)

    return tf.abs(z)


def normalize(x):
    x -= tf.reduce_min(x, axis=SPATIAL_AXES, keepdims=True)
    x /= tf.reduce_max(x, axis=SPATIAL_AXES, keepdims=True)

    return x


@tf.function
def validate(
        x: tf.Tensor
) -> tf.Tensor:
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    return x


def save(
        image: tf.Tensor,
        output: str,
):
    if image.shape[0] == 1: image = image[0, ...]
    if image.shape[-1] == 1: image = image[..., 0]

    im = Image.fromarray(image.numpy())
    im.save(output)


# endregion


def main(
        path: str,
        output: str,
        low_pass: int,
        high_pass: int,
        band_pass: Tuple[int, int],
        compression_rate: float,
        rotation: int,
        ext: str = '.png',
):
    images = load_images(path)

    os.makedirs(output, exist_ok=True)

    for file_name, image in images:
        print(f'Processing input image {file_name}')
        print(f'  shape: {image.shape}')
        print(f'  size: {np.prod(image.shape) * 4 / 10 ** 6:.1f} MB')

        basename = os.path.splitext(os.path.basename(file_name))[0]

        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)

        names = (
            f'identity',
            f'low-{low_pass}',
            f'high-{high_pass}',
            f'band-{band_pass}',
            f'compressed-{compression_rate}',
            f'rotated-{rotation}')

        results = transform(image, low_pass, high_pass, band_pass, compression_rate, rotation)

        for k, (y, f) in zip(names, results):
            dst = os.path.join(output, f'{basename}-{k}')
            save(validate(y), f'{dst}{ext}')
            save(validate(255 * normalize(tf.math.log(tf.abs(f) + 1))), f'{dst}-filter{ext}')
            print(f'  {dst} saved')


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('-i', '--images', required=True, help='input image or directory of images to be processed')
    p.add_argument('-o', '--output', required=True, help='path to results folder')

    p.add_argument('--low-pass', default=10, type=int, help='low-pass filter radius (in frequency)')
    p.add_argument('--high-pass', default=30, type=int, help='high-pass filter radius (in frequency)')
    p.add_argument('--band-pass', default=(10, 30), type=int, nargs='+', help='Band-pass filter radius (in frequency)')
    p.add_argument('--compression-rate',
                   default=95,
                   type=float,
                   help='percentile of magnitudes that should be suppressed during compression')
    p.add_argument('--rotation', default=45, type=int, help='image rotation (in degrees)')

    args = p.parse_args()

    try:
        main(
            args.images,
            args.output,
            args.low_pass,
            args.high_pass,
            tuple(args.band_pass),
            args.compression_rate,
            args.rotation,
        )

    except Exception as e:
        print('An error has occurred during processing:', file=sys.stderr)
        print(str(e), file=sys.stderr)
        exit(1)

    except KeyboardInterrupt:
        print('interrupted')
        exit(2)
