"""Process images using commmon Filters.

Author: Lucas David -- <lucas.david@ic.unicamp.br>

"""

import os
import pathlib
from argparse import ArgumentParser
from math import ceil, floor
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

tf.get_logger().setLevel('ERROR')

_PADDINGS = ('VALID', 'SAME')

# region kernels

SEPIA = tf.constant(
    [[0.393, 0.349, 0.272],
     [0.769, 0.686, 0.534],
     [0.189, 0.168, 0.131]]
)

GRAY = tf.constant([
    0.2989, 0.5870, 0.1140
])

H17 = tf.transpose(
    tf.constant(
        [
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]],

            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]],

            [[-1., -1., -1.],
             [-1., 8., -1.],
             [-1., -1., -1.]],

            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]],

            [[-1., -1., 2.],
             [-1., 2., -1.],
             [2., -1., -1.]],

            [[2., -1., -1.],
             [-1., 2., -1.],
             [-1., -1., 2.]],

            [[0., 0., 1.],
             [0., 0., 0.],
             [-1., 0., 0.]],
        ]
    ),
    (1, 2, 0)
)

H89 = tf.transpose(
    tf.stack((
        [[0., 0., -1., 0., 0.],
         [0., -1., -2., -1., 0.],
         [-1., -2., 16., -2., -1.],
         [0., -1., -2., -1., 0.],
         [0., 0., -1., 0., 0.]],
        1 / 256 * tf.constant(
            [[1., 4., 6., 4., 1.],
             [4., 16., 24., 16., 4.],
             [6., 24., 36., 24., 6.],
             [4., 16., 24., 16., 4.],
             [1., 4., 16., 4., 1.]]
        )), axis=0),
    (1, 2, 0)
)


# endregion

# region exercises

@tf.function
def sepia(
        x: tf.Tensor
) -> tf.Tensor:
    return x @ SEPIA


@tf.function
def grayscale(
        x: tf.Tensor
) -> tf.Tensor:
    return tf.tensordot(x, GRAY, 1)


def correlate2d(
        s,
        k,
        padding='VALID'
):
    s, k = map(np.asarray, (s, k))
    _validate_correlate2d_args(s, k, padding)

    B, H, W = s.shape
    KH, KW, KC = k.shape

    if padding == 'SAME':
        pt, pb = floor((KH - 1) / 2), ceil((KH - 1) / 2)
        pl, pr = floor((KW - 1) / 2), ceil((KW - 1) / 2)

        s = np.pad(s, ((0, 0), (pt, pb), (pl, pr)))
        B, H, W = s.shape

    # Creating selection tile s[0:3, 0:3]
    #   --> [s[0,0], s[0,1], s[0,2], s[1,0], s[1,1], s[1,2]]
    r0 = np.arange(H - KH + 1)
    r0 = np.repeat(r0, W - KW + 1)
    r0 = r0.reshape(-1, 1)

    r1 = np.arange(KH).reshape(1, KH)
    r = np.repeat(r0 + r1, KW, axis=1)

    c0 = np.arange(KW)
    c0 = np.tile(c0, KH).reshape(1, -1)

    c1 = np.arange(W - KW + 1).reshape(-1, 1)
    c = c0 + c1
    c = np.tile(c, [H - KH + 1, 1])

    # k.shape (3, 3) --> (9, 1), in order to multiply
    # and add-reduce in a single pass with "@".
    y = s[..., r, c] @ k.reshape(-1, KC)
    y = y.reshape(B, H - KH + 1, W - KW + 1, KC)

    return y


def convolve2d(s, k, padding='VALID'):
    k = np.rot90(k, k=2)  # reflect signal.
    return correlate2d(s, k, padding)


# endregion

# region Util functions

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


def _validate_correlate2d_args(s, k, padding):
    assert padding in _PADDINGS, (f'Unknown value {padding} for argument `padding`. '
                                  f'It must be one of the following: {_PADDINGS}')
    assert len(s.shape) == 3, (f'Input `s` must have shape [B, H, W]. '
                               f'A tensor of shape {s.shape} was passed.')
    assert len(k.shape) == 3, (f'Kernels `k` must have shape [H, W, C]. '
                               f'A tensor of shape {k.shape} was passed.')


# endregion


def main(
        path: str,
        output: str,
        padding: str = 'SAME',
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
            for transform in (sepia, grayscale):
                dst = os.path.join(output, f'{basename}-{transform.__name__}{ext}')
                save(validate(transform(image)), dst)
                print(f'  {dst} saved')
        else:
            image = image[tf.newaxis, ..., 0]

            kernels = 'h1 h2 h3 h4 h5 h6 h7 h8 h9 sqrt(h1^2+h2^2)'.split()
            results = many_convolutions(image, padding)

            for k, y in zip(kernels, results):
                dst = os.path.join(output, f'{basename}-{k}{ext}')
                save(y, dst)
                print(f'  {dst} saved')


def many_convolutions(images, padding):
    return [
        *flat_images_and_kernels(validate(convolve2d(images, H17, padding=padding))),
        *flat_images_and_kernels(validate(convolve2d(images, H89, padding=padding))),
        *flat_images_and_kernels(validate(np.sqrt(convolve2d(images, H17[..., 0:1], padding=padding) ** 2 +
                                                  convolve2d(images, H17[..., 1:2], padding=padding) ** 2)))
    ]


def flat_images_and_kernels(x):
    return tf.reshape(tf.transpose(x, (0, 3, 1, 2)), (-1, *x.shape[1:3]))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('-i', '--images', required=True, help='input image or directory of images to be processed')
    p.add_argument('-o', '--output', required=True, help='path to results folder')
    p.add_argument('-p', '--padding', default='SAME', help='padding used when processing monochromatic images',
                   choices=_PADDINGS)

    args = p.parse_args()

    # try:
    main(
        args.images,
        args.output,
        args.padding.upper(),
    )
    # except Exception as e:
    #   print('An error has occurred during processing:', file=sys.stderr)
    #   print(str(e), file=sys.stderr)
    #   exit(1)

    # except KeyboardInterrupt:
    #   print('interrupted')
    #   exit(2)
