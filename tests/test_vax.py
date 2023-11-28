################################################################################
# tests/test_vax.py
################################################################################

import numpy as np
import os
import unittest
import sys
from vax import from_vax32, to_vax32, from_vax64


PYTHON2 = sys.version_info.major <= 2


class Test_Vax(unittest.TestCase):

  def test_vax32(self):

    np.random.seed(4744)

    BIGINT = 2**24      # all conversions should be good to at least 24 bits
    SCALE = 1. / BIGINT
    EXPMIN = -125
    EXPMAX = 127

    # Single-value inversion tests
    reals = []
    complexes = []
    for k in range(1000):
        mantissa = np.random.randint(-BIGINT, BIGINT) * SCALE
        exponent = np.random.randint(EXPMIN, EXPMAX)
        real_ = mantissa * 2.**exponent
        reals.append(real_)

        mantissa = np.random.randint(-BIGINT, BIGINT) * SCALE
        exponent = np.random.randint(EXPMIN, EXPMAX)
        comp = complex(real_, mantissa * 2.**exponent)
        complexes.append(comp)

        for ieee, type_, dtype in [(real_, np.float32,   np.dtype('<f4')),
                                   (comp,  np.complex64, np.dtype('<c8'))]:

            # Scalar
            result = from_vax32(to_vax32(ieee))
            self.assertEqual(result, ieee)
            self.assertTrue(np.isscalar(result))
            self.assertIsInstance(result, type_)

            # Shapeless array
            arg = np.array(ieee, dtype=dtype)
            result = from_vax32(to_vax32(arg))
            self.assertEqual(result, arg)
            self.assertFalse(np.isscalar(result))
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(result.shape == ())
            self.assertTrue(result.dtype == dtype)

            # Array of shape (1,)
            arg = np.array([ieee], dtype=dtype)
            result = from_vax32(to_vax32(arg))
            self.assertTrue(np.all(result == arg))
            self.assertFalse(np.isscalar(result))
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(result.shape == (1,))
            self.assertTrue(result.dtype == dtype)

            # Array of shape (1,1)
            arg = np.array([[ieee]], dtype=dtype)
            result = from_vax32(to_vax32(arg))
            self.assertTrue(np.all(result == arg))
            self.assertFalse(np.isscalar(result))
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(result.shape == (1,1))
            self.assertTrue(result.dtype == dtype)

            # Array-like of shape (1,)
            result = from_vax32(to_vax32([ieee]))
            self.assertTrue(np.all(result == arg))
            self.assertFalse(np.isscalar(result))
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(result.shape == (1,))
            self.assertTrue(result.dtype == dtype)

            # Array-like of shape (1,1)
            result = from_vax32(to_vax32([[ieee]]))
            self.assertTrue(np.all(result == arg))
            self.assertFalse(np.isscalar(result))
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(result.shape == (1,1))
            self.assertTrue(result.dtype == dtype)

    # Array-like inversion tests
    results = from_vax32(to_vax32(reals))
    self.assertTrue(np.all(results == reals))

    results = from_vax32(to_vax32(complexes))
    self.assertTrue(np.all(results == complexes))

    test = [[reals[   :125], reals[125:250], reals[250:375], reals[375:500]],
            [reals[500:625], reals[625:750], reals[750:875], reals[875:]]]
    results = from_vax32(to_vax32(test))
    self.assertTrue(np.all(results == test))
    self.assertEqual(results.shape, (2,4,125))

    test = list(range(10))
    self.assertEqual(test, list(from_vax32(to_vax32(test))))

    # Array inversion tests
    for k in range(10):
        for shape in [(7,), (7,7), (7,7,7), (4,1), (4,2), (4,3),
                      (3,), (3,1), (3,1,1), (1,3), (1,3,1), (1,1,3),
                      (1,3,1,1,1,1)]:
            mantissa = np.random.randint(-BIGINT, BIGINT, size=shape) * SCALE
            exponent = np.random.randint(EXPMIN, EXPMAX, size=shape)
            real_ = mantissa * 2.**exponent

            cshape = shape + (2,)
            mantissa = np.random.randint(-BIGINT, BIGINT, size=cshape) * SCALE
            exponent = np.random.randint(EXPMIN, EXPMAX, size=cshape)
            temp = mantissa * 2.**exponent
            comp = temp.view('complex').reshape(shape)

            for ieee, dtype in [(real_, np.dtype('<f4')),
                                (comp,  np.dtype('<c8'))]:
                result = from_vax32(to_vax32(ieee))
                self.assertTrue(isinstance(result, np.ndarray))
                self.assertTrue(result.shape == ieee.shape)
                self.assertTrue(result.dtype == dtype)
                self.assertTrue(np.all(result == ieee))

    # Single-value buffer, memoryview, bytes, bytearray, str
    for k in range(100):
        mantissa = np.random.randint(-BIGINT, BIGINT) * SCALE
        exponent = np.random.randint(EXPMIN, EXPMAX)
        ieee = mantissa * 2.**exponent

        # buffer (Python 2) or memoryview (Python 3)
        vax32 = to_vax32(ieee)
        result = from_vax32(vax32.data)
        self.assertTrue(np.isscalar(result))
        self.assertEqual(result, ieee)

        # bytes
        result = from_vax32(bytes(vax32.data))
        self.assertTrue(np.isscalar(result))
        self.assertEqual(result, ieee)

        # bytearray
        result = from_vax32(bytearray(vax32.data))
        self.assertTrue(np.isscalar(result))
        self.assertEqual(result, ieee)

        # string
        if PYTHON2:
            result = from_vax32(str(vax32.data))    # pragma: no cover
        else:
            result = from_vax32(str(vax32.data, encoding='latin8'))
        self.assertTrue(np.isscalar(result))
        self.assertEqual(result, ieee)

    # Multiple-value buffer, memoryview, bytes, bytearray, str
    for k in range(100):
        size = np.random.randint(2,21)
        mantissa = np.random.randint(-BIGINT, BIGINT, size=size) * SCALE
        exponent = np.random.randint(EXPMIN, EXPMAX, size=size)
        ieee = (mantissa * 2.**exponent).astype('<f4')

        # buffer (Python 2) or memoryview (Python 3)
        vax32 = to_vax32(ieee)
        result = from_vax32(vax32.data)
        self.assertTrue(result.shape == ieee.shape)
        self.assertTrue(np.all(result == ieee))

        # bytes
        result = from_vax32(bytes(vax32.data))
        self.assertTrue(result.shape == ieee.shape)
        self.assertTrue(np.all(result == ieee))

        # bytearray
        result = from_vax32(bytearray(vax32.data))
        self.assertTrue(result.shape == ieee.shape)
        self.assertTrue(np.all(result == ieee))

        # string
        if PYTHON2:
            result = from_vax32(str(vax32.data))    # pragma: no cover
        else:
            result = from_vax32(str(vax32.data, encoding='latin8'))
        self.assertTrue(result.shape == ieee.shape)
        self.assertTrue(np.all(result == ieee))

    # Try some real-world Vax data from
    #   VGISS_5xxx/VGISS_5214/CALIB/MIPL/VGRSCF.DAT
    #
    # array = np.fromfile('.../VGRSCF.DAT', dtype='uint8')[780:]

    uints = np.array([
       104,  64,  39,  49,  96,  64, 156, 196,  60,  64,   8, 172, 128,
        64,   0,   0, 125,  64,  27,  47, 128,  64,   0,   0, 131,  64,
        10, 215,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0, 110,  64,   4,  86,  77,  64,
       242, 210, 128,  64,   0,   0, 134,  64,  25,   4, 110,  64,   4,
        86, 132,  64,  88,  57, 132,  64,  88,  57,   8,  65, 176, 114,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,  32,  54,  47,  49,  53,  47,  56,  53,
       128,  64,   0,   0, 136,  64,  57, 180, 130,  64, 229, 208, 101,
        64, 203, 161, 128,  64,   0,   0, 128,  64,  78,  98, 128,  64,
         0,   0, 125,  64,  27,  47,  86,  65, 123,  20, 100,  65, 166,
       155,  90,  65, 131, 192,  64,  65,   0,   0, 128,  64,   0,   0,
        86,  65,  82, 184,  86,  65, 123,  20,  83,  65,  70, 182,  88,
        66,  53,  94, 103,  66, 111,  18,  93,  66, 184,  30,  66,  66,
       123,  20, 128,  64,   0,   0,  89,  66,  12,   2,  88,  66,  53,
        94,  85,  66, 231, 251,  88,  66,  53,  94, 103,  66, 111,  18,
        93,  66, 184,  30,  66,  66, 123,  20, 128,  64,   0,   0,  89,
        66,  12,   2,  88,  66,  53,  94,  85,  66, 231, 251, 105,  64,
        94, 186,  98,  64, 211,  77, 128,  64,   0,   0, 118,  64,  25,
         4, 105,  64,  94, 186, 129,  64, 252, 169, 129,  64, 252, 169,
        73,  65, 252, 169,  67,  65, 188, 116,  61,  65, 125,  63,  86,
        65, 123,  20,  77,  65, 143, 194,  67,  65, 188, 116,  88,  65,
       254, 212,  88,  65, 254, 212,  40,  66, 215, 163,  69,  66,  55,
       137,  63,  66, 150,  67,  88,  66,  53,  94,  79,  66, 231, 251,
        69,  66,  55, 137,  91,  66,   2,  43,  91,  66,   2,  43,  42,
        67, 170, 113,  69,  66,  55, 137,  63,  66, 150,  67,  88,  66,
        53,  94,  79,  66, 231, 251,  69,  66,  55, 137,  91,  66,   2,
        43,  91,  66,   2,  43,  42,  67, 170, 113, 128,  64,   0,   0,
       104,  64,  39,  49,  96,  64, 156, 196,  60,  64,   8, 172, 128,
        64,   0,   0, 125,  64,  27,  47, 128,  64,   0,   0, 131,  64,
        10, 215,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0, 110,  64,   4,  86,  77,  64,
       242, 210, 128,  64,   0,   0, 134,  64,  25,   4, 110,  64,   4,
        86, 132,  64,  88,  57, 132,  64,  88,  57,   8,  65, 176, 114,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,  32,  50,  47,  48,  50,  47,  56,  54],
      dtype='uint8')

    ieee = from_vax32(uints)
    truth = np.array([1.   , 1.068, 1.022, 0.897, 1.   , 1.003, 1.   , 0.989])
    self.assertTrue(np.all(truth.astype('<f4') == ieee[65:73]))

    truth = np.array([3.345, 3.572, 3.418, 3.   , 1.   , 3.355, 3.345, 3.308])
    self.assertTrue(np.all(truth.astype('<f4') == ieee[73:81]))

    truth = np.array([13.523, 14.442, 13.82 , 12.13 ,  1.   , 13.563, 13.523,
                      13.374])
    self.assertTrue(np.all(truth.astype('<f4') == ieee[81:89]))
    self.assertTrue(np.all(truth.astype('<f4') == ieee[89:97]))

    ieee = from_vax32(uints.reshape(39,20))
    self.assertEqual(ieee.shape, (39,5))

    ieee = from_vax32(uints.reshape(39,5,4))
    self.assertEqual(ieee.shape, (39,5))

    self.assertRaises(ValueError, from_vax32, uints[:9]) # not a multiple of 4

    # Errors
    self.assertRaises(ValueError, from_vax32, b'1')
    self.assertRaises(ValueError, from_vax32, np.arange(10, dtype='f8'))
    self.assertRaises(ValueError, from_vax32, np.array(['123', '456']))
    self.assertRaises(ValueError, from_vax32, ['123', '456'])

    # Reading from a file...
    vax_dir = os.path.split(sys.modules['vax'].__file__)[0]
    parent = os.path.split(vax_dir)[0]
    test_file = os.path.join(parent, 'test_files', 'C3490702_GEOMA.DAT')
    data = np.fromfile(test_file, dtype='<f4')
    data = data[1536//4:]       # skip LBLSIZE
    data = from_vax32(data)
    data = data.reshape(-1,4)   # four columns

    answer_10 = np.array([
        [ 25.36      ,  25.31      ,   9.831727  ,  15.862788  ],
        [ 25.36      ,  25.31      ,   9.831727  ,  15.862788  ],
        [ 20.53      ,  85.59      ,   2.6002007 ,  60.86287   ],
        [ 20.53      ,  85.59      ,   2.6002007 ,  60.86287   ],
        [ 25.36      , 177.6       ,   3.0882928 , 133.30754   ],
        [ 25.36      , 177.6       ,   3.0882928 , 133.30754   ],
        [ 25.36      , 269.74      ,   1.6376413 , 208.92227   ],
        [ 25.36      , 269.74      ,   1.6376413 , 208.92227   ],
        [ 25.36      , 362.04      ,   0.71304405, 285.84558   ],
        [ 25.36      , 362.04      ,   0.71304405, 285.84558   ]], dtype='float32')
    self.assertTrue(np.all(answer_10 == data[:10]))

    # Tests copied from libvaxdata
    vax_f4 = np.array(
        [0x80, 0x40, 0x00, 0x00,
         0x80, 0xC0, 0x00, 0x00,
         0x60, 0x41, 0x00, 0x00,
         0x60, 0xC1, 0x00, 0x00,
         0x49, 0x41, 0xD0, 0x0F,
         0x49, 0xC1, 0xD0, 0x0F,
         0xF0, 0x7D, 0xC2, 0xBD,
         0xF0, 0xFD, 0xC2, 0xBD,
         0x08, 0x03, 0xEA, 0x1C,
         0x08, 0x83, 0xEA, 0x1C,
         0x9E, 0x40, 0x52, 0x06,
         0x9E, 0xC0, 0x52, 0x06],
        dtype='uint8').reshape(-1,4)

    ieee_f4 = np.array([
       1.000000    ,
      -1.000000    ,
       3.500000    ,
      -3.500000    ,
       3.141590    ,
      -3.141590    ,
      9.9999999E+36,
     -9.9999999E+36,
      9.9999999E-38,
     -9.9999999E-38,
       1.234568    ,
      -1.234568    ], dtype='float32')

    diff = np.abs((from_vax32(vax_f4) - ieee_f4) / ieee_f4)
    self.assertEqual(np.median(diff), 0.)
    self.assertTrue(np.all(diff < 1.e-7))
    answer = from_vax32(vax_f4)

    self.assertTrue(np.all(from_vax32(bytes(vax_f4.data)) == answer))

    if not PYTHON2:
        string = bytes(vax_f4.data).decode('latin8')
        self.assertTrue(np.all(from_vax32(string) == answer))

    ints = vax_f4.view('<i2')
    self.assertTrue(np.all(from_vax32(ints) == answer))

    ints = vax_f4.view('>i2')
    self.assertTrue(np.all(from_vax32(ints) == answer))

    ints = vax_f4.view('<i4').reshape(-1)
    self.assertTrue(np.all(from_vax32(ints) == answer))

    ints = vax_f4.view('>i4').reshape(-1)
    self.assertTrue(np.all(from_vax32(ints) == answer))

    floats = vax_f4.view('<f4').reshape(-1)
    self.assertTrue(np.all(from_vax32(floats) == answer))

    floats = vax_f4.view('>f4').reshape(-1)
    self.assertTrue(np.all(from_vax32(floats) == answer))

    comps = vax_f4.reshape(-1,8).view('<c8').reshape(-1)
    ieee_c8 = answer.view('<c8')
    self.assertTrue(np.all(from_vax32(comps) == ieee_c8))

    comps = vax_f4.reshape(-1,8).view('>c8').reshape(-1)
    self.assertTrue(np.all(from_vax32(comps) == ieee_c8))

    floats = vax_f4.view('=f4').reshape(-1)
    for k,x in enumerate(floats):
        self.assertEqual(from_vax32(x), answer[k])

    comps = vax_f4.reshape(-1,8).view('=c8').reshape(-1)
    for k,x in enumerate(comps):
        self.assertEqual(from_vax32(x), ieee_c8[k])


  def test_vax64(self):

    # Tests copied from libvaxdata
    vax_d8 = np.array(
        [0x80, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x80, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x60, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x60, 0xC1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x49, 0x41, 0xDA, 0x0F, 0x21, 0xA2, 0xBE, 0x68,
         0x49, 0xC1, 0xDA, 0x0F, 0x21, 0xA2, 0xBE, 0x68,
         0xF0, 0x7D, 0xC2, 0xBD, 0xBB, 0x1A, 0xDB, 0x48,
         0xF0, 0xFD, 0xC2, 0xBD, 0xBB, 0x1A, 0xDB, 0x48,
         0x08, 0x03, 0xEA, 0x1C, 0x54, 0x14, 0x75, 0x5C,
         0x08, 0x83, 0xEA, 0x1C, 0x54, 0x14, 0x75, 0x5C,
         0x9E, 0x40, 0x52, 0x06, 0x62, 0x14, 0xE7, 0xCE,
         0x9E, 0xC0, 0x52, 0x06, 0x62, 0x14, 0xE7, 0xCE],
        dtype='uint8').reshape(-1,8)

    ieee_f8 = np.array([
        1.000000000000000    ,
       -1.000000000000000    ,
        3.500000000000000    ,
       -3.500000000000000    ,
        3.141592653589793    ,
       -3.141592653589793    ,
       1.0000000000000000E+37,
      -1.0000000000000000E+37,
       9.9999999999999999E-38,
      -9.9999999999999999E-38,
        1.234567890123450    ,
       -1.234567890123450    ], dtype='float64')

    diff = np.abs((from_vax64(vax_d8) - ieee_f8) / ieee_f8)
    self.assertTrue(np.all(from_vax64(vax_d8) == ieee_f8))

    self.assertTrue(np.all(from_vax64(vax_d8.data) == ieee_f8))

    self.assertTrue(np.all(from_vax64(bytes(vax_d8.data)) == ieee_f8))

    if not PYTHON2:
        string = bytes(vax_d8.data).decode('latin8')
        self.assertTrue(np.all(from_vax64(string) == ieee_f8))

    ints = vax_d8.view('<i2')
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    ints = vax_d8.view('>i2')
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    ints = vax_d8.view('<i4')
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    ints = vax_d8.view('>i4')
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    ints = vax_d8.view('<i8').reshape(-1)
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    ints = vax_d8.view('>i8').reshape(-1)
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    floats = vax_d8.view('<f8').reshape(-1)
    self.assertTrue(np.all(from_vax64(floats) == ieee_f8))

    floats = vax_d8.view('>f8').reshape(-1)
    self.assertTrue(np.all(from_vax64(floats) == ieee_f8))

    comps = vax_d8.reshape(-1,16).view('<c16').reshape(-1)
    ieee_c16 = ieee_f8.view('<c16')
    self.assertTrue(np.all(from_vax64(comps) == ieee_c16))

    comps = vax_d8.reshape(-1,16).view('>c16').reshape(-1)
    self.assertTrue(np.all(from_vax64(comps) == ieee_c16))

    floats = vax_d8.view('=f8').reshape(-1)
    for k,x in enumerate(floats):
        self.assertEqual(from_vax64(x), ieee_f8[k])

    for k,x in enumerate(floats):
        self.assertEqual(from_vax64(np.array(x)), np.array(ieee_f8[k]))

    comps = vax_d8.reshape(-1,16).view('=c16').reshape(-1)
    for k,x in enumerate(comps):
        self.assertEqual(from_vax64(x), ieee_c16[k])

    ints = vax_d8.view('>i2').reshape(-1,4)
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8))

    ints = vax_d8.view('>i2').reshape(-1,8)
    self.assertTrue(np.all(from_vax64(ints) == ieee_f8.reshape(-1,2)))

    # Errors
    self.assertRaises(ValueError, from_vax64, b'1')
    self.assertRaises(ValueError, from_vax64, np.array(['123', '456']))
    self.assertRaises(ValueError, from_vax64, ['123', '456'])
    self.assertRaises(ValueError, from_vax64, vax_d8.reshape(-1)[:7])
    self.assertRaises(ValueError, from_vax64, vax_d8[:,:4])

##########################################################################################
# Perform unit testing if executed from the command line
##########################################################################################

if __name__ == "__main__":
    unittest.main()     # pragma: no cover

##########################################################################################
