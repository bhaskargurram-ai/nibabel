"""Benchmarks for writing gzip files with zlib and ISA-L

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_openers.py
"""

import os
import sys
from time import perf_counter
from unittest import mock

import numpy as np

from .. import Nifti1Image
from .._compression import HAVE_ISAL
from ..openers import Opener
from ..tmpdirs import InTemporaryDirectory
from .butils import print_git_title


def _best_of(func, repeat):
    times = []
    for _ in range(repeat):
        tic = perf_counter()
        func()
        times.append(perf_counter() - tic)
    return min(times)


def bench_gzip_write():
    rng = np.random.default_rng(20261104)
    repeat = 3
    # A smooth "brain" with noise and a zero background, so the data compress
    # like a real image; the shape is a quarter of a 2mm MNI bold series
    img_shape = (46, 55, 46, 60)
    center = np.array(img_shape[:3], dtype=np.float32) / 2
    grid = np.indices(img_shape[:3], dtype=np.float32)
    dist2 = ((grid - center[:, None, None, None]) ** 2).sum(axis=0)
    brain = np.exp(-dist2 / 200).astype(np.float32) * 1000
    arr = brain[..., None] + rng.normal(0, 20, img_shape).astype(np.float32)
    arr[brain < 50] = 0
    raw = arr.tobytes()
    img = Nifti1Image(arr, np.eye(4))

    sys.stdout.flush()
    print()
    print_git_title('gzip write')
    print(f'{len(raw) / 1e6:.1f} MB raw float32, best of {repeat}')
    fmt = '{:36s} {:7.3f} s {:7.1f} MB'.format

    backends = [('zlib', False)]
    if HAVE_ISAL:
        backends.append(('ISA-L', True))
    with InTemporaryDirectory():
        for name, use_isal in backends:
            with mock.patch('nibabel._compression.USE_ISAL', use_isal):
                for level in (1, 3):

                    def write_level():
                        with Opener('bench.gz', 'wb', compresslevel=level) as fobj:
                            fobj.write(raw)

                    mtime = _best_of(write_level, repeat)
                    size = os.path.getsize('bench.gz') / 1e6
                    print(fmt(f'Opener write, {name} level {level}', mtime, size))

                mtime = _best_of(lambda: img.to_filename('bench.nii.gz'), repeat)
                size = os.path.getsize('bench.nii.gz') / 1e6
                print(fmt(f'Nifti1Image.to_filename, {name}', mtime, size))
    sys.stdout.flush()
