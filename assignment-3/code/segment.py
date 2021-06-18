"""Image Segmentation.

Author: Lucas David -- <lucas.david@ic.unicamp.br>

"""

import os
import pathlib
from argparse import ArgumentParser
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage import (io,
                     color,
                     filters,
                     segmentation)
from skimage.measure import regionprops_table

DEFAULT_IMAGE_EXT = '.png'

BORDERS = {
  'sobel': filters.sobel,
  'roberts': filters.roberts,
  'prewitt': filters.prewitt,
  'farid': filters.farid,
  'scharr': filters.scharr
}

REGION_PROPERTIES = (
  'label',
  'area',
  'convex_area',
  'eccentricity',
  'solidity',
  'perimeter',
  'centroid',
  'bbox')

COLORS = sns.color_palette("Set3")


# region exercises

def extract_borders(
    image,
    method,
):
  borders = BORDERS[method](image)
  borders = 1 - borders  # white bg

  return borders


def extract_segment(image):
  return segmentation.felzenszwalb(image, scale=1e6, sigma=0.1, min_size=10)


def print_properties_report(ps, name=None, detail=None):
  if name: print(name)

  print('número de regiões:', len(ps))
  print('número de regiões pequenas:', (ps.area < 1500).sum())
  print('número de regiões grandes:', ((ps.area >= 1500) & (ps.area < 3000)).sum())
  print('número de regiões grandes:', (ps.area >= 3000).sum())
  print()

  if detail:
    ps = ps.head(detail)

  for ix, p in ps.iterrows():
    print(f'região {p.label:6.0f}: área: {int(p.area):6d} perímetro: {p.perimeter:12.6f} '
          f'excentricidade: {p.eccentricity:.6f} solidez: {p.solidity:.6f}')

  print()


# endregion

# region Util functions

def load_images(
    path: str
) -> List[Tuple[str, np.ndarray]]:
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

  if os.path.isfile(path):
    files = [path]
  else:
    files = list(map(str, pathlib.Path(path).glob('*')))

  images = [(f, io.imread(f)) for f in files]
  print(f'{len(images)} found')

  return images


def validate(
    x: np.ndarray
) -> np.ndarray:
  return np.clip(255 * x, 0, 255).astype('uint8')


def save(
    result: Union[np.ndarray, Image.Image, pd.DataFrame],
    output: str,
):
  if isinstance(result, np.ndarray):
    if result.shape[0] == 1: result = result[0, ...]
    if result.shape[-1] == 1: result = result[..., 0]

    result = Image.fromarray(result)
    result.save(output + DEFAULT_IMAGE_EXT)
  elif isinstance(result, plt.Figure):
    result.savefig(output + DEFAULT_IMAGE_EXT)
  elif isinstance(result, pd.DataFrame):
    result.to_csv(output + '.csv', index=False)


def plot_segments(g, s, p, alpha=0.8, linewidth=1):
  fig = plt.figure()
  plt.imshow(color.label2rgb(s, image=g, bg_label=0, alpha=alpha, colors=COLORS, bg_color=(1, 1, 1)))
  plt.axis('off')
  ax = plt.gca()

  for i, region in p.iterrows():
    minr, minc, maxr, maxc = region['bbox-0'], region['bbox-1'], region['bbox-2'], region['bbox-3']
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False,
                         edgecolor=COLORS[i % len(COLORS)], linewidth=linewidth)
    ax.add_patch(rect)

    plt.text(region['centroid-1'], region['centroid-0'], int(region.label))

  plt.tight_layout()

  return fig


def plot_object_size_histogram(props):
  fig = plt.figure()
  sns.histplot(props.area)
  plt.tight_layout()

  return fig


# endregion


def extract_properties(segment):
  return pd.DataFrame(regionprops_table(segment, properties=REGION_PROPERTIES))


def transform(image, file_name, borders_method):
  gray = color.rgb2gray(image)
  borders = extract_borders(gray, borders_method)
  segments = extract_segment(image)
  props = extract_properties(segments)

  print_properties_report(props)

  return {
    'gray': validate(gray),
    'borders': validate(borders),
    'segmentation': plot_segments(image, segments, props),
    'size_histogram': plot_object_size_histogram(props),
    'properties': props
  }


def main(
    path: str,
    output: str,
    borders_method: str,
):
  images = load_images(path)

  os.makedirs(output, exist_ok=True)

  for file_name, image in images:
    print(f'Processing input image {file_name}')
    print(f'  shape: {image.shape}')
    print(f'  size: {np.prod(image.shape) * 4 / 10 ** 6:.1f} MB')

    basename = os.path.splitext(os.path.basename(file_name))[0]

    results = transform(image, file_name, borders_method)

    for k, r in results.items():
      dst = os.path.join(output, f'{basename}-{k}')
      save(r, dst)
      print(f'  {dst} saved')


if __name__ == '__main__':
  p = ArgumentParser(description=__doc__)
  p.add_argument('-i', '--images', required=True, help='input image or directory of images to be processed')
  p.add_argument('-o', '--output', required=True, help='path to results folder')
  p.add_argument('-b', '--borders', choices=BORDERS.keys(), default='scharr', help='border extraction method')

  args = p.parse_args()

  try:
    main(args.images, args.output, args.borders)
  except Exception as e:
    print('An error has occurred during processing:', file=sys.stderr)
    print(str(e), file=sys.stderr)
    exit(1)
  except KeyboardInterrupt:
    print('interrupted')
    exit(2)
