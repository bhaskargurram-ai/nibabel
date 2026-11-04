# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Constants and types for dealing transparently with compression"""

from __future__ import annotations

import bz2
import gzip
import io
import typing as ty

try:
    from compression import zstd  # type: ignore[import-not-found]

    HAVE_ZSTD = True
except ImportError:  # PY313
    HAVE_ZSTD = False

from .deprecated import alert_future_error
from .optpkg import optional_package

if ty.TYPE_CHECKING:
    import indexed_gzip  # type: ignore[import]
    from isal import igzip as isal_igzip

    if not HAVE_ZSTD:  # PY313
        from backports import zstd  # type: ignore[import]

        HAVE_ZSTD = True

    HAVE_INDEXED_GZIP = True
    HAVE_ISAL = True
    HAVE_ZSTD = True

    ModeRT = ty.Literal['r', 'rt']
    ModeRB = ty.Literal['rb']
    ModeWT = ty.Literal['w', 'wt']
    ModeWB = ty.Literal['wb']
    ModeR = ModeRT | ModeRB
    ModeW = ModeWT | ModeWB
    Mode = ModeR | ModeW

else:
    indexed_gzip, HAVE_INDEXED_GZIP, _ = optional_package('indexed_gzip')
    isal_igzip, HAVE_ISAL, _ = optional_package('isal.igzip')
    if not HAVE_ZSTD:  # PY313
        zstd, HAVE_ZSTD, _ = optional_package('backports.zstd')

# Collections of types for isinstance or exception matching
COMPRESSED_FILE_LIKES: tuple[type[io.IOBase], ...] = (
    bz2.BZ2File,
    gzip.GzipFile,
)
COMPRESSION_ERRORS: tuple[type[BaseException], ...] = (
    OSError,  # BZ2File
    gzip.BadGzipFile,
)

if HAVE_INDEXED_GZIP:
    COMPRESSED_FILE_LIKES += (indexed_gzip.IndexedGzipFile,)
    COMPRESSION_ERRORS += (indexed_gzip.ZranError,)
    from indexed_gzip import IndexedGzipFile  # type: ignore[import-not-found]
else:
    IndexedGzipFile = gzip.GzipFile

if HAVE_ZSTD:
    COMPRESSED_FILE_LIKES += (zstd.ZstdFile,)
    COMPRESSION_ERRORS += (zstd.ZstdError,)

#: Write gzip files with ISA-L (the ``isal`` package) when it is installed.
#:
#: ISA-L produces standard gzip streams several times faster than zlib.
#: Set to ``False`` to always write with the standard library :mod:`gzip`
#: module, for example to get byte-identical output to an older setup::
#:
#:     import nibabel
#:     nibabel._compression.USE_ISAL = False
#:
#: This has no effect when ``isal`` is not installed (see :data:`HAVE_ISAL`).
USE_ISAL: bool = True

#: Compression levels that are handed to ISA-L.
#:
#: ISA-L only implements levels 0-3, and its level 0 still compresses, unlike
#: zlib's level 0, which stores the data uncompressed. To keep the meaning of
#: ``compresslevel`` unchanged, level 0 and levels 4-9 are written with the
#: standard library :mod:`gzip` module.
ISAL_COMPRESSLEVELS: range = range(1, 4)


def _isal_for_level(compresslevel: int | None) -> bool:
    """True if ISA-L is installed, enabled, and implements ``compresslevel``"""
    return HAVE_ISAL and USE_ISAL and compresslevel in ISAL_COMPRESSLEVELS


def _normalize_gzip_args(
    filename: str | None,
    mode: Mode | None,
    fileobj: io.FileIO | None,
) -> tuple[str, io.FileIO, io.FileIO | None]:
    """Return a binary mode string, a file object, and the file object we opened (if any)"""
    if mode is None:
        mode = 'rb'
    modestr: str = mode

    # These two guards are adapted from
    # https://github.com/python/cpython/blob/6ab65c6/Lib/gzip.py#L171-L174
    if 'b' not in modestr:
        modestr = f'{mode}b'
    myfileobj = None
    if fileobj is None:
        if filename is None:
            raise TypeError('Must define either fileobj or filename')
        # Cast because GzipFile.myfileobj has type io.FileIO while open returns ty.IO
        fileobj = myfileobj = ty.cast('io.FileIO', open(filename, modestr))
    return modestr, fileobj, myfileobj


class DeterministicGzipFile(gzip.GzipFile):
    """Deterministic variant of GzipFile

    This writer does not add filename information to the header, and defaults
    to a modification time (``mtime``) of 0 seconds.
    """

    def __init__(
        self,
        filename: str | None = None,
        mode: Mode | None = None,
        compresslevel: int | None = None,
        fileobj: io.FileIO | None = None,
        mtime: int = 0,
    ):
        modestr, fileobj, myfileobj = _normalize_gzip_args(filename, mode, fileobj)
        if myfileobj is not None:
            self.myfileobj = myfileobj
        # Only forward compresslevel when explicitly set; None defers to Python's runtime default
        compress_kwargs: dict[str, int] = {}
        if compresslevel is not None:
            compress_kwargs['compresslevel'] = compresslevel
        super().__init__(
            filename='',
            mode=modestr,
            fileobj=fileobj,
            mtime=mtime,
            **compress_kwargs,
        )


if HAVE_ISAL:

    class DeterministicIGzipFile(isal_igzip.IGzipFile):
        """Deterministic variant of ``isal.igzip.IGzipFile``

        This writer uses the ISA-L library to produce standard gzip streams.
        Like :class:`DeterministicGzipFile`, it does not add filename information
        to the header, and defaults to a modification time (``mtime``) of 0 seconds.

        ISA-L only implements compression levels 0-3; ``compresslevel`` must be
        one of these, and the default is 1 (the :class:`~nibabel.openers.Opener`
        default). Open files for writing only.
        """

        def __init__(
            self,
            filename: str | None = None,
            mode: Mode | None = None,
            compresslevel: int | None = None,
            fileobj: io.FileIO | None = None,
            mtime: int = 0,
        ):
            if compresslevel is None:
                compresslevel = 1
            modestr, fileobj, myfileobj = _normalize_gzip_args(filename, mode, fileobj)
            if myfileobj is not None:
                self.myfileobj = myfileobj
            super().__init__(
                filename='',
                mode=modestr,
                compresslevel=compresslevel,
                fileobj=fileobj,
                mtime=mtime,
            )


def gzip_open(
    filename: str,
    mode: Mode = 'rb',
    compresslevel: int | None = None,
    mtime: int = 0,
    keep_open: bool = False,
) -> gzip.GzipFile:
    """Open a gzip file for reading or writing.

    If opening a file for reading, and ``indexed_gzip`` is available,
    an ``IndexedGzipFile`` is returned.

    If opening a file for writing, ``isal`` is available and enabled
    (see :data:`USE_ISAL`), and ``compresslevel`` is one of
    :data:`ISAL_COMPRESSLEVELS`, a ``DeterministicIGzipFile`` is returned,
    which writes with the ISA-L library.

    Otherwise, a ``DeterministicGzipFile`` is returned.

    Parameters:
    -----------

    filename : str
        Path of file to open.
    mode : str
        Opening mode - either ``rb`` or ``wb``.
    compresslevel: int | None
        Compression level when writing. If None, the running Python's default is used.
    mtime: int
        Modification time used when writing a file - passed to the
        ``DeterministicGzipFile``. Ignored when reading.
    keep_open: bool
        Whether to keep the file handle open between reads. Ignored when writing,
        or when ``indexed_gzip`` is not present.
    """
    gzip_file: gzip.GzipFile
    if not HAVE_INDEXED_GZIP or mode != 'rb':
        # use ISA-L for faster writes, when the level is one it implements
        if 'w' in mode and _isal_for_level(compresslevel):
            gzip_file = DeterministicIGzipFile(filename, mode, compresslevel, mtime=mtime)
        else:
            gzip_file = DeterministicGzipFile(filename, mode, compresslevel, mtime=mtime)

    # use indexed_gzip if possible for faster read access.  If keep_open ==
    # True, we tell IndexedGzipFile to keep the file handle open. Otherwise
    # the IndexedGzipFile will close/open the file on each read.
    else:
        gzip_file = IndexedGzipFile(filename, drop_handles=not keep_open)

    return gzip_file


def zstd_open(
    filename: str,
    mode: Mode = 'r',
    *,
    level: int | None = None,
    options: dict | None = None,
    zstd_dict: zstd.ZstdDict | None = None,
    level_or_option: int | dict | None = None,
) -> zstd.ZstdFile:
    """Open a zstd file for reading or writing.

    The specific object returned will be a ``compression.zstd.ZstdFile`` or
    a ``backports.zstd.ZstdFile``.

    Parameters
    ----------

    filename : str
        Path of file to open.
    mode : str
        Opening mode.
    zstd_dict : ZstdDict
        Dictionary used for compression/decompression.
    level : int
        Compression level when writing.
    options : dict
        Dictionary of compression/decompression options.
    """
    if level_or_option is not None:
        alert_future_error(
            'The level_or_option parameter will be removed in a future version of nibabel',
            '7.0',
            warning_rec='This warning can be silenced by using the separate level/option parameters',
            error_rec='Future errors can be avoided by using the separate level/option parameters',
            error_class=TypeError,
        )
        if level is not None or options is not None:
            raise ValueError('Only one of level_or_option, level or options may be specified')
        if isinstance(level_or_option, int):
            level = level_or_option
        else:
            options = level_or_option
    return zstd.ZstdFile(filename, mode, level=level, options=options, zstd_dict=zstd_dict)
