# cython: language_level=3
# distutils: language=c
from math import ceil
from io import BytesIO
from typing import Union, Dict, BinaryIO, Tuple

from libc.stdint cimport uint32_t

from cpython.ref cimport PyObject
import numpy as np
cimport numpy as cnp

cdef extern struct JPEG2000Parameters:
    uint32_t columns
    uint32_t rows
    int colourspace
    uint32_t nr_components
    uint32_t precision
    unsigned int is_signed
    uint32_t nr_tiles

cdef extern char* OpenJpegVersion()
cdef extern int Decode(void* fp, unsigned char* out, int codec)
cdef extern int GetParameters(void* fp, int codec, JPEG2000Parameters *param)
cdef extern int Encode(
    cnp.PyArrayObject* arr,
    void *dst,
    int codec_format,
    int bits_stored,
    int photometric_interpretation,
    int lossless,
    int use_mct,
    int compression_ratio,
)


ERRORS = {
    1: "failed to create the input stream",
    2: "failed to setup the decoder",
    3: "failed to read the header",
    4: "failed to set the component indices",
    5: "failed to set the decoded area",
    6: "failed to decode image",
    7: "support for more than 16-bits per component is not implemented",
    8: "failed to upscale subsampled components",
}
ENCODING_ERRORS = {
    1: (
        "the input array has an invalid shape, must be (rows, columns) or "
        "(rows, columns, planes)"
    ),
    2: "the input array has an invalid number of samples per pixel, must be 1, 3 or 4",
    3: "the input array has an invalid dtype, only bool, u1, u2, i1 and i2 are supported",
    4: "the input array must use little endian byte ordering",
    5: "the input array has an invalid shape, the number of rows must be in (1, 65535)",
    6: "the input array has an invalid shape, the number of columns must be in (1, 65535)",
    7: "the input array must be C-style, contiguous and aligned",
    8: "the image precision given by bits stored must be in (1, itemsize of the input array's dtype)"
}


def get_version() -> bytes:
    """Return the openjpeg version as bytes."""
    cdef char *version = OpenJpegVersion()

    return version


def decode(
    fp: BinaryIO,
    codec: int = 0,
    as_array: bool = False
) -> Union[np.ndarray, bytearray]:
    """Return the decoded JPEG 2000 data from Python file-like `fp`.

    Parameters
    ----------
    fp : file-like
        A Python file-like containing the encoded JPEG 2000 data. Must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    codec : int, optional
        The codec to use for decoding, one of:

        * ``0``: JPEG-2000 codestream
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format
    as_array : bool, optional
        If ``True`` then return the decoded image data as a :class:`numpy.ndarray`
        otherwise return the data as a :class:`bytearray` (default).

    Returns
    -------
    bytearray | numpy.ndarray
        If `as_array` is False (default) then returns the decoded image data
        as a :class:`bytearray`, otherwise returns the image data as a
        :class:`numpy.ndarray`.

    Raises
    ------
    RuntimeError
        If unable to decode the JPEG 2000 data.
    """
    param = get_parameters(fp, codec)
    bpp = ceil(param['precision'] / 8)
    nr_bytes = param['rows'] * param['columns'] * param['nr_components'] * bpp

    cdef PyObject* p_in = <PyObject*>fp
    cdef unsigned char *p_out
    if as_array:
        out = np.zeros(nr_bytes, dtype=np.uint8)
        p_out = <unsigned char *>cnp.PyArray_DATA(out)
    else:
        out = bytearray(nr_bytes)
        p_out = <unsigned char *>out

    result = Decode(p_in, p_out, codec)
    if result != 0:
        raise RuntimeError(
            f"Error decoding the J2K data: {ERRORS.get(result, result)}"
        )

    return out


def get_parameters(fp: BinaryIO, codec: int = 0) -> Dict[str, Union[str, int, bool]]:
    """Return a :class:`dict` containing the JPEG 2000 image parameters.

    Parameters
    ----------
    fp : file-like
        A Python file-like containing the encoded JPEG 2000 data.
    codec : int, optional
        The codec to use for decoding, one of:

        * ``0``: JPEG-2000 codestream
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format

    Returns
    -------
    dict
        A :class:`dict` containing the J2K image parameters:
        ``{'columns': int, 'rows': int, 'colourspace': str,
        'nr_components: int, 'precision': int, `is_signed`: bool,
        'nr_tiles: int'}``. Possible colour spaces are "unknown",
        "unspecified", "sRGB", "monochrome", "YUV", "e-YCC" and "CYMK".

    Raises
    ------
    RuntimeError
        If unable to decode the JPEG 2000 data.
    """
    cdef JPEG2000Parameters param
    param.columns = 0
    param.rows = 0
    param.colourspace = 0
    param.nr_components = 0
    param.precision = 0
    param.is_signed = 0
    param.nr_tiles = 0

    # Pointer to the JPEGParameters object
    cdef JPEG2000Parameters *p_param = &param

    # Pointer to J2K data
    cdef PyObject* ptr = <PyObject*>fp

    # Decode the data - output is written to output_buffer
    result = GetParameters(ptr, codec, p_param)
    if result != 0:
        try:
            msg = f": {ERRORS[result]}"
        except KeyError:
            pass

        raise RuntimeError("Error decoding the J2K data" + msg)

    # From openjpeg.h#L309
    colours = {
        -1: "unknown",
         0: "unspecified",
         1: "sRGB",
         2: "monochrome",
         3: "YUV",
         4: "e-YCC",
         5: "CYMK",
    }

    try:
        colourspace = colours[param.colourspace]
    except KeyError:
        colourspace = "unknown"

    parameters = {
        'rows' : param.rows,
        'columns' : param.columns,
        'colourspace' : colourspace,
        'nr_components' : param.nr_components,
        'precision' : param.precision,
        'is_signed' : bool(param.is_signed),
        'nr_tiles' : param.nr_tiles,
    }

    return parameters


def encode(
    cnp.ndarray arr,
    int codec,
    int bits_stored,
    int photometric_interpretation,
    int lossless,
    int use_mct,
    int compression_ratio,
) -> Tuple[int, bytes]:
    # The source for the image data
    cdef cnp.PyArrayObject* p_in = <cnp.PyArrayObject*>arr
    # The destination for the encoded J2K codestream
    dst = BytesIO()
    p_out = <void *>dst

    result = Encode(
        p_in,
        p_out,
        codec,
        bits_stored,
        photometric_interpretation,
        lossless,
        use_mct,
        compression_ratio,
    )

    # TODO: just return, let Python handle raises exceptions etc
    if result != 0:
        raise RuntimeError(
            f"Error encoding the data: {ENCODING_ERRORS.get(result, result)}"
        )

    return result, dst.getvalue()
