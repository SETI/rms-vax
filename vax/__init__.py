"""PDS Ring-Moon Systems Node, SETI Institute
Module "vax"

Functions to convert between Vax single-precision floats and IEEE floats.

from_vax32  Interpret this byte string, array, or array-like as Vax float4 or
            complex8 values and return the equivalent single-precision IEEE
            value(s).

to_vax32    Convert this number, array, or array-like into an array of Vax
            float4 or complex8 values with the same shape.

to_vax32_bytes
            Convert this number, array, or array-like into a byte string
            containing the binary representation of the equivalent Vax
            float4 or complex8 value(s).

This module continues to support both Python 2 in addition to Python 3.
"""

import numpy as np
import sys

try:
    from _version import __version__
except ImportError as err:
    __version__ = 'Version unspecified'

_PYTHON2 = sys.version_info.major <= 2


def from_vax32(data):
    """Interpret this byte string, array, or array-like as Vax float4 or
    complex8 values and return the equivalent single-precision IEEE value(s).

    If the input is an array, the shape of that array is preserved except for
    the last axis, which may be modified account for the new itemsize.

    If the input array is complex, the returned array will have dtype "<c8";
    otherwise, it will have dtype "<f4".
    """

    # Convert a string to bytes; also handle a Python 2 buffer
    if _PYTHON2:
        if isinstance(data, (str, buffer)):     # pragma: no cover
            data = bytes(data)
    else:
        if isinstance(data, str):
            data = bytes(data, encoding='latin8')

    # Convert the object to a NumPy array with an even number of 2-byte elements
    if isinstance(data, (bytes, bytearray, memoryview)):
        nbytes = data.nbytes if isinstance(data, memoryview) else len(data)
        if nbytes % 4 != 0:
            raise ValueError('data size is not a multiple of 4 bytes')

        array = np.frombuffer(data, dtype='<f4')
        scalar = (nbytes == 4)          # True to convert to scalar at the end
        shapeless = False
        newshape = (nbytes // 4,)       # array shape after conversion
        dtype = '<f4'

    else:
        scalar = np.isscalar(data)      # True to return a scalar
        array = np.asarray(data, order='C')
        shapeless = array.shape == ()   # True to convert back to shape ()

        array = np.atleast_1d(array)    # needed for the view to work below

        # Validate array and array-like data types; convert to LSB
        if isinstance(data, np.ndarray):
            key = array.dtype.kind + str(array.dtype.itemsize)
            if key not in {'f4', 'c8', 'u1', 'u2', 'u4', 'i1', 'i2', 'i4'}:
                raise ValueError('invalid data type for array input: '
                                 + str(array.dtype))
            array = np.asarray(array, dtype='<' + key)
            dtype = '<c8' if key == 'c8' else '<f4'

        else:
            # Conversion of array-like produces arrays with dtype "f8" or "c16"
            if array.dtype.kind == 'c':
                array = np.asarray(array, dtype='<c8')
                dtype = '<c8'
            elif array.dtype.kind in 'uif':
                array = np.asarray(array, dtype='<' + array.dtype.kind + '4')
                dtype = '<f4'
            else:
                raise ValueError('invalid data type for array-like input: '
                                 + str(array.dtype))

        # Determine array shape after conversion
        if array.itemsize in (1,2):
            if (array.shape[-1] * array.itemsize) % 4 != 0:
                raise ValueError('last axis size is not a multiple of 4 bytes')

            last_axis = (array.shape[-1] * array.itemsize) // 4
            if last_axis == 1:
                newshape = array.shape[:-1]
            else:
                newshape = array.shape[:-1] + (last_axis,)

        else:
            newshape = array.shape

    itemsize = 8 if dtype == '<c8' else 4

    # Conversion involves a pairwise swap of bytes and then division by 4
    pairs = array.view(dtype='<u2')
    pairs = pairs.reshape(-1, 2)
    swapped = pairs[:,::-1].copy()
    swapped = swapped.reshape(-1, itemsize//2)
    ieee = swapped.view(dtype=dtype) / 4.

    if scalar:
        return ieee[0,0]                # current shape is (1,1)
    elif shapeless:
        return ieee.reshape(())
    else:
        return ieee.reshape(newshape)


def to_vax32_bytes(array):
    """Convert this number, array, or array-like into a byte string containing
    the binary representation of the equivalent Vax float4 or complex8 value(s).
    """

    # Make array contiguous, with C index order, containing 4-byte IEEE floats
    dtype = '<c8' if np.iscomplexobj(array) else '<f4'
    array = np.asarray(array, dtype=dtype, order='C')

    # Conversion involves multiplication by 4 and then a pairwise byte swap
    paired_view = (4. * np.atleast_1d(array)).view('<u2')
    paired_view = paired_view.reshape(-1, 2)
    swapped = paired_view[:,::-1].copy()

    return swapped.tobytes()


def to_vax32(array):
    """Convert this number, array, or array-like into an array of Vax float4 or
    complex8 values with the same shape.

    If the input is a scalar, the returned object is an array of shape ().

    If the input is complex, the returned array will be of dtype "<c8";
    otherwise, it will be of dtype "<f4".

    Note that this object will not be usable for arithmetic operations in its
    returned form.
    """

    # Make array contiguous, with C index order, containing 4-byte IEEE floats
    dtype = '<c8' if np.iscomplexobj(array) else '<f4'
    scalar = np.isscalar(array)
    array = np.asarray(array, dtype=dtype, order='C')

    # Construct array from converted buffer
    buffer = to_vax32_bytes(array)
    result = np.frombuffer(buffer, dtype=dtype).reshape(array.shape)

    if scalar:
        return result[()]
    else:
        return result

################################################################################
