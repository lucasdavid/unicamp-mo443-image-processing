"""Texture.

Author: Lucas David -- <lucas.david@ic.unicamp.br>

"""

import os
import pathlib
import sys
from argparse import ArgumentParser
from math import ceil
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.spatial.distance import cdist
from skimage import io, color, feature
from skimage.feature import greycomatrix, greycoprops


class Config:
  class lbp:
    p = 8
    r = 1
    method = 'default'

  class glcm:
    distances = [1, 2, 3, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    levels = 256

  default_image_ext = '.png'


COLORS = sns.color_palette("Set3")


# region metrics and properties

def kullback_leibler_divergence(p, q):
  """Kullback-Leibler Divergence.

  :ref: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
  """
  p = np.asarray(p)
  q = np.asarray(q)
  filt = np.logical_and(p != 0, q != 0)
  return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def bhattacharyya(a, b):
  """ Bhattacharyya distance between distributions.

  :ref: https://gist.github.com/miku/1671b2014b003ee7b9054c0618c805f7
  """
  return -np.log(np.sqrt(a * b).sum())


METRICS = {
  'kld': kullback_leibler_divergence,
  'kullback_leibler_divergence': kullback_leibler_divergence,
  'bhattacharyya': bhattacharyya,
  'braycurtis': 'braycurtis',
  'canberra': 'canberra',
  'chebyshev': 'chebyshev',
  'cityblock': 'cityblock',
  'correlation': 'correlation',
  'cosine': 'cosine',
  'dice': 'dice',
  'euclidean': 'euclidean',
  'hamming': 'hamming',
  'jaccard': 'jaccard',
  'jensenshannon': 'jensenshannon',
  'kulsinski': 'kulsinski',
  'mahalanobis': 'mahalanobis',
  'matching': 'matching',
  'minkowski': 'minkowski',
  'rogerstanimoto': 'rogerstanimoto',
  'russellrao': 'russellrao',
  'seuclidean': 'seuclidean',
  'sokalmichener': 'sokalmichener',
  'sokalsneath': 'sokalsneath',
  'sqeuclidean': 'sqeuclidean',
  'wminkowski': 'wminkowski',
  'yule': 'yule'
}

# GREYCO_PROPERTIES = 'contrast,dissimilarity,homogeneity,energy,correlation,ASM'
GREYCO_PROPERTIES = 'dissimilarity,correlation'


# endregion


# region exercises

def extract_lbp(
    image,
    p: int = None,
    r: float = None,
    method: str = None,
):
  return feature.local_binary_pattern(
    image,
    P=p or Config.lbp.p,
    R=r or Config.lbp.r,
    method=method or Config.lbp.method)


def extract_glcm_features(
    image,
    properties: str = None,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = None,
):
  image = (image * 255).astype('uint8')

  cm = greycomatrix(
    image,
    distances=distances or Config.glcm.distances,
    angles=angles or Config.glcm.angles,
    levels=levels or Config.glcm.levels,
    symmetric=True,
    normed=True)

  props = properties or GREYCO_PROPERTIES
  props = props.split(',')
  features = np.asarray([greycoprops(cm, prop=prop) for prop in props]).ravel()

  return features


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
    x: np.ndarray,
    scale=255,
) -> np.ndarray:
  return np.clip(scale * x, 0, 255).astype('uint8')


def save(
    result: Union[np.ndarray, Image.Image, pd.DataFrame],
    output: str,
):
  if isinstance(result, np.ndarray):
    if result.shape[0] == 1: result = result[0, ...]
    if result.shape[-1] == 1: result = result[..., 0]

    result = Image.fromarray(result)
    result.save(output + Config.default_image_ext)
  elif isinstance(result, plt.Figure):
    result.savefig(output + Config.default_image_ext)
  elif isinstance(result, pd.DataFrame):
    result.to_csv(output + '.csv', index=False)


def visualize(
    image,
    title=None,
    rows=2,
    cols=None,
    cmap=None,
    figsize=(12, 6)
):
  if image is not None:
    cols = cols or ceil(len(image) / rows)

    if isinstance(image, (list, tuple)) or len(image.shape) > 3:  # many images
      fig = plt.figure(figsize=figsize)
      for ix in range(len(image)):
        plt.subplot(rows, cols, ix + 1)
        visualize(image[ix],
                  cmap=cmap,
                  title=title[ix] if title is not None and len(title) > ix else None,
                  rows=rows,
                  cols=cols)
      plt.tight_layout()
      return fig

    if image.shape[-1] == 1: image = image[..., 0]
    plt.imshow(image, cmap=cmap)

  if title is not None: plt.title(title)
  plt.axis('off')


def plot_histogram(lbp):
  n_bins = int(lbp.max() + 1)

  fig = plt.figure()
  plt.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')
  plt.tight_layout()

  return fig


def plot_similarity_matrix(d, names):
  d_normed = d - d.min(axis=0, keepdims=True)
  d_normed /= d.max(axis=0, keepdims=True)

  with sns.axes_style("white"):
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(d_normed, annot=d, cmap="RdPu_r", cbar=False, annot_kws={'size': 8},
                xticklabels=names, yticklabels=names, mask=np.tril(d) <= 0)
    plt.tight_layout()
    return fig


def plot_most_similar(images, names, dst):
  similarity_index = np.argsort(dst, axis=1)[:, 1:4]
  ds = np.take_along_axis(dst, similarity_index, axis=1)

  vis = sum([[i] + [images[j] for j in s]
             for i, s in zip(images, similarity_index)], [])

  titles = sum([[f'{anchor_label} (anchor)'] +
                [f'{names[ix]} d:{d:.2f}' for ix, d in zip(ii, di)]
                for ii, di, anchor_label in zip(similarity_index, ds, names)
                ], [])

  return visualize(
    vis,
    titles,
    rows=len(images),
    figsize=(12, 2 * len(images)))


# endregion


def transform(image, file_name, metric, glcm_props):
  gray = color.rgb2gray(image)
  lbp = extract_lbp(gray)
  bins = int(lbp.max() + 1)

  glcm_features = extract_glcm_features(gray, glcm_props)

  return {
    'gray': validate(gray),
    'lbp': validate(lbp, scale=1),
    'lbp_h': plot_histogram(lbp),
    'lbp_features': np.histogram(lbp, density=True, bins=bins, range=(0, bins))[0],
    'glcm_features': glcm_features
  }


def match(names, images, lbps, glcms, metric):
  dh = cdist(lbps, lbps, metric)
  dm = cdist(glcms, glcms, metric)

  smh = plot_similarity_matrix(dh, names)
  smm = plot_similarity_matrix(dm, names)

  msh = plot_most_similar(images, names, dh)
  msm = plot_most_similar(images, names, dm)

  return {
    'lbp_similarity': smh,
    'glcm_similarity': smm,
    'lbp_matches': msh,
    'glcm_matches': msm
  }


def main(
    path: str,
    output: str,
    metric: str,
    glcm_props: str
):
  images = load_images(path)

  os.makedirs(output, exist_ok=True)

  report = {'file_name': [], 'image': [], 'lbp': [], 'glcm': []}

  for file_name, image in images:
    print(f'Processing input image {file_name}')
    print(f'  shape: {image.shape}')
    print(f'  size: {np.prod(image.shape) * 4 / 10 ** 6:.1f} MB')

    report['file_name'].append(file_name)
    report['image'].append(image)

    basename = os.path.splitext(os.path.basename(file_name))[0]

    results = transform(image, file_name, metric, glcm_props)

    for k, r in results.items():
      if k.endswith('features'):
        report[k.rstrip('_features')].append(r)
      else:
        dst = os.path.join(output, f'{basename}-{k}')
        save(r, dst)
        print(f'  {dst} saved')

  results = match(
    [os.path.basename(n) for n in report['file_name']],
    report['image'],
    report['lbp'],
    report['glcm'],
    metric)

  os.makedirs(os.path.join(output, 'matching'), exist_ok=True)

  for k, r in results.items():
    dst = os.path.join(output, 'matching', k)
    save(r, dst)
    print(f'{dst} saved')


if __name__ == '__main__':
  p = ArgumentParser(description=__doc__)
  p.add_argument('-i', '--images', required=True, help='input image or directory of images to be processed')
  p.add_argument('-o', '--output', required=True, help='path to results folder')
  p.add_argument('-d', '--distance', choices=METRICS.keys(), default='kld',
                 help='distance metric used when comparing LBP histograms')
  p.add_argument('-p', '--glcm-props', default=GREYCO_PROPERTIES, help='Properties used when comparing GLCM features')

  args = p.parse_args()

  try:
    main(args.images, args.output, args.distance, args.props)
  except Exception as e:
    print('An error has occurred during processing:', file=sys.stderr)
    print(str(e), file=sys.stderr)
    exit(1)
  except KeyboardInterrupt:
    print('interrupted')
    exit(2)
