
from copy import copy
from io import BytesIO
from math import ceil
from pathlib import Path
import warnings

import numpy as np

import _openjpeg


class EncodingOptions:
    """
    """
    def __init__(self):
        """Create a new EncodingOptions instance.

        Parameters
        ----------
        arr : numpy.ndarray, optional

        """
        self.subsampling = (1, 1)
        self.progression_order = 0
        self.block_size = (64, 64)
        self.lossless = True
        self.codec = 0  # J2K

        # Input array parameters
        self.width = None
        self.height = None
        self.nr_components = None
        self.precision = None
        self.is_signed = False

    # NOTE: it might be easier to just pass the 0x0028 elements + UID
    def from_pixel_data(self, arr, uid, bit_depth=Non):
        """Setup the encoding options based on a `uid`

        Parameters
        ----------
        arr : numpy.ndarray
            The array containing the pixel data to be encoded.
        uid : pydicom.uid.UID
            The *Transfer Syntax UID* to use for encoding. Currently only
            supports *JPEG 2000 Lossless*.
        bit_depth : int, optional
            If the array contains N-bit data (where N is not a multiple of 8),
            then this should be N. For example, a uint16 numpy array may
            only contain 12-bit data, in which case you should pass
            ``bit_depth = 12``.
        """
        shape = arr.shape
        if len(shape) > 3:
            raise ValueError("The input array must be a single frame only")
        elif len(shape) == 2:
            self.nr_components = 1
        else:
            self.nr_components = shape[2]

        self.height = shape[0]
        self.width = shape[1]

        if self.nr_components not in [1, 3]:
            raise ValueError("The input array must have 1 or 3 channels only")

        dtype = arr.dtype
        if not np.issubdtype(dtype, np.integer):
            raise ValueError("The input array must be made up of integers")

        self.is_signed = np.issubdtype(dtype, np.signedinteger)
        self.precision = dtype.itemsize * 8
        if bit_depth:
            self.precision = bit_depth

        if uid == '1.2.840.10008.1.2.4.90':
            self.subsampling = (1, 1)
            self.progression_order = 0
            self.block_size = (64, 64)
            self.lossless = True
            self.codec = 0

            return

        # TODO: Add support for lossy
        # Hmm, how do we handle the quality?
        if uid == '1.2.840.10008.1.2.4.91':
            self.subsampling = (1, 1)
            self.progression_order = 0
            self.block_size = (64, 64)
            self.lossless = False
            self.codec = 0

        raise NotImplementedError(
            f"JPEG 2000 encoding for the Transfer Syntax UID '{uid}' is "
            "not supported"
        )

    def to_dict(self):
        opts = {}
        # Encoding parameters
        opts['dx'] = self.subsampling[0]
        opts['dy'] = self.subsampling[1]
        opts['prog_order'] = self.progression_order
        opts['cblockw_init'] = self.block_size[0]
        opts['cblockh_init'] = self.block_size[1]
        opts['irreversible'] = int(not self.lossless)
        opts['cod_format'] = self.codec

        # Image parameters
        opts['h'] = self.height
        opts['w'] = self.width
        opts['numcomps'] = self.nr_components
        opts['prec'] = opts['bpp'] = self.precision
        opts['sgnd'] = int(self.is_signed)

        return opts

    def validate(self):
        pass


def _get_format(stream):
    """Return the JPEG 2000 format for the encoded data in `stream`.

    Parameters
    ----------
    stream : bytes or file-like
        A Python object containing the encoded JPEG 2000 data. If not
        :class:`bytes` then the object must have ``tell()``, ``seek()`` and
        ``read()`` methods.

    Returns
    -------
    int
        The format of the encoded data, one of:

        * ``0``: JPEG-2000 codestream
        * ``2``: JP2 file format

    Raises
    ------
    ValueError
        If no matching JPEG 2000 file format found for the data.
    """
    data = stream.read(20)
    stream.seek(0)
    #print(" ".join([f"{ii:02X}" for ii in data[:12]]))

    magic_numbers = {
        # JPEG 2000 codestream, has no header, .j2k, .jpc, .j2c
        b"\xff\x4f\xff\x51": 0,
        # JP2 and JP2 RFC3745, .jp2
        b"\x0d\x0a\x87\x0a": 2,
        b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a": 2,
        # JPT, .jpt - shrug
    }

    try:
        return magic_numbers[data[:4]]
    except KeyError:
        pass

    try:
        return magic_numbers[data[:12]]
    except KeyError:
        pass

    raise ValueError("No matching JPEG 2000 format found")


def get_openjpeg_version():
    """Return the openjpeg version as tuple of int."""
    version = _openjpeg.get_version().decode("ascii").split(".")
    return tuple([int(ii) for ii in version])


def decode(stream, j2k_format=None, reshape=True):
    """Return the decoded JPEG2000 data from `stream` as a
    :class:`numpy.ndarray`.

    .. versionchanged:: 1.1

        `stream` can now also be :class:`str` or :class:`pathlib.Path`

    Parameters
    ----------
    stream : str, pathlib.Path, bytes or file-like
        The path to the JPEG 2000 file or a Python object containing the
        encoded JPEG 2000 data. If using a file-like then the object must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    j2k_format : int, optional
        The JPEG 2000 format to use for decoding, one of:

        * ``0``: JPEG-2000 codestream (such as from DICOM *Pixel Data*)
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format
    reshape : bool, optional
        Reshape and re-view the output array so it matches the image data
        (default), otherwise return a 1D array of ``np.uint8``.

    Returns
    -------
    numpy.ndarray
        An array of containing the decoded image data.

    Raises
    ------
    RuntimeError
        If the decoding failed.
    """
    if isinstance(stream, (str, Path)):
        with open(stream, 'rb') as f:
            stream = f.read()

    if isinstance(stream, (bytes, bytearray)):
        stream = BytesIO(stream)

    required_methods = ["read", "tell", "seek"]
    if not all([hasattr(stream, meth) for meth in required_methods]):
        raise TypeError(
            "The Python object containing the encoded JPEG 2000 data must "
            "either be bytes or have read(), tell() and seek() methods."
        )

    if j2k_format is None:
        j2k_format = _get_format(stream)

    if j2k_format not in [0, 1, 2]:
        raise ValueError(f"Unsupported 'j2k_format' value: {j2k_format}")

    arr = _openjpeg.decode(stream, j2k_format)
    if not reshape:
        return arr

    meta = get_parameters(stream, j2k_format)
    bpp = ceil(meta["precision"] / 8)

    dtype = f"uint{8 * bpp}" if not meta["is_signed"] else f"int{8 * bpp}"
    arr = arr.view(dtype)

    shape = [meta["rows"], meta["columns"]]
    if meta["nr_components"] > 1:
        shape.append(meta["nr_components"])

    return arr.reshape(*shape)


def decode_pixel_data(stream, ds=None):
    """Return the decoded JPEG 2000 data as a :class:`numpy.ndarray`.

    Intended for use with *pydicom* ``Dataset`` objects.

    Parameters
    ----------
    stream : bytes or file-like
        A Python object containing the encoded JPEG 2000 data. If not
        :class:`bytes` then the object must have ``tell()``, ``seek()`` and
        ``read()`` methods.
    ds : pydicom.dataset.Dataset, optional
        A :class:`~pydicom.dataset.Dataset` containing the group ``0x0028``
        elements corresponding to the *Pixel data*. If used then the
        *Samples per Pixel*, *Bits Stored* and *Pixel Representation* values
        will be checked against the JPEG 2000 data and warnings issued if
        different.

    Returns
    -------
    numpy.ndarray
        A 1D array of ``numpy.uint8`` containing the decoded image data.

    Raises
    ------
    RuntimeError
        If the decoding failed.
    """
    if isinstance(stream, (bytes, bytearray)):
        stream = BytesIO(stream)

    required_methods = ["read", "tell", "seek"]
    if not all([hasattr(stream, meth) for meth in required_methods]):
        raise TypeError(
            "The Python object containing the encoded JPEG 2000 data must "
            "either be bytes or have read(), tell() and seek() methods."
        )

    j2k_format = _get_format(stream)
    if j2k_format != 0:
        warnings.warn(
            "The (7FE0,0010) Pixel Data contains a JPEG 2000 codestream "
            "with the optional JP2 file format header, which is "
            "non-conformant to the DICOM Standard (Part 5, Annex A.4.4)"
        )

    arr = _openjpeg.decode(stream, j2k_format)

    if not ds:
        return arr

    meta = get_parameters(stream, j2k_format)
    if ds.SamplesPerPixel != meta["nr_components"]:
        warnings.warn(
            f"The (0028,0002) Samples per Pixel value '{ds.SamplesPerPixel}' "
            f"in the dataset does not match the number of components "
            f"\'{meta['nr_components']}\' found in the JPEG 2000 data. "
            f"It's recommended that you change the  Samples per Pixel value "
            f"to produce the correct output"
        )

    if ds.BitsStored != meta["precision"]:
        warnings.warn(
            f"The (0028,0101) Bits Stored value '{ds.BitsStored}' in the "
            f"dataset does not match the component precision value "
            f"\'{meta['precision']}\' found in the JPEG 2000 data. "
            f"It's recommended that you change the Bits Stored value to "
            f"produce the correct output"
        )

    if bool(ds.PixelRepresentation) != meta["is_signed"]:
        val = "signed" if meta["is_signed"] else "unsigned"
        ds_val = "signed" if bool(ds.PixelRepresentation) else "unsigned"
        ds_val = f"'{ds.PixelRepresentation}' ({ds_val})"
        warnings.warn(
            f"The (0028,0103) Pixel Representation value {ds_val} in the "
            f"dataset does not match the format of the values found in the "
            f"JPEG 2000 data '{val}'"
        )

    return arr


def encode_pixel_data(arr, uid):
    """Return JPEG encoded pixel data.

    .. versionadded:: 1.2

    **JPEG 2000 Lossless**

    * Lossless encoding (reversible DWT 5-3)
    * 1 tile
    * If 3 channel RGB then YCC colourspace conversion will be applied prior
      to encoding (? check this)
    * No subsampling
    * Precinct size 2**15 by 2**15
    * Code-block size: 64 by 64
    * No SOP or EPH markers in the codestream
    * Progression order: LRCP
    * No JP2 header

    Parameters
    ----------
    arr : numpy.ndarray
        The pixel data to be encoded, must be either 1 channel monochrome or
        3 channel RGB.
    uid : pydicom.uid.UID
        The *Transfer Syntax UID* that the pixel data is to be encoded into.
        Currently only *JPEG Lossless* is supported.

    Returns
    -------
    bytes
        The JPEG encoded pixel data.
    """
    if uid != '1.2.840.10008.1.2.4.90':
        raise NotImplementedError(
            "Only 'JPEG 2000 Lossless' is currently supported"
        )

    opts = EncodingOptions()
    opts.from_pixel_data(arr, uid)
    opts.validate()

    return _openjpeg.encode(arr, opts.to_dict())


def get_parameters(stream, j2k_format=None):
    """Return a :class:`dict` containing the JPEG2000 image parameters.

    .. versionchanged:: 1.1

        `stream` can now also be :class:`str` or :class:`pathlib.Path`

    Parameters
    ----------
    stream : str, pathlib.Path, bytes or file-like
        The path to the JPEG 2000 file or a Python object containing the
        encoded JPEG 2000 data. If using a file-like then the object must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    j2k_format : int, optional
        The JPEG 2000 format to use for decoding, one of:

        * ``0``: JPEG-2000 codestream (such as from DICOM *Pixel Data*)
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format

    Returns
    -------
    dict
        A :class:`dict` containing the J2K image parameters:
        ``{'columns': int, 'rows': int, 'colourspace': str,
        'nr_components: int, 'precision': int, `is_signed`: bool}``. Possible
        colour spaces are "unknown", "unspecified", "sRGB", "monochrome",
        "YUV", "e-YCC" and "CYMK".

    Raises
    ------
    RuntimeError
        If reading the image parameters failed.
    """
    if isinstance(stream, (str, Path)):
        with open(stream, 'rb') as f:
            stream = f.read()

    if isinstance(stream, (bytes, bytearray)):
        stream = BytesIO(stream)

    required_methods = ["read", "tell", "seek"]
    if not all([hasattr(stream, func) for func in required_methods]):
        raise TypeError(
            "The Python object containing the encoded JPEG 2000 data must "
            "either be bytes or have read(), tell() and seek() methods."
        )

    if j2k_format is None:
        j2k_format = _get_format(stream)

    if j2k_format not in [0, 1, 2]:
        raise ValueError(f"Unsupported 'j2k_format' value: {j2k_format}")

    return _openjpeg.get_parameters(stream, j2k_format)
