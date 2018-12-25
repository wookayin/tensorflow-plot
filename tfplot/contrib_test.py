'''Unit Test for tfplot.contrib'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os
import scipy.misc

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # filter out INFO and WARN logs
tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib
matplotlib.rcParams['figure.figsize'] = (2.5, 2.5)

from termcolor import cprint
from imgcat import imgcat

import tfplot.contrib
import tfplot.test_util as test_util


class TestContrib(test_util.TestcaseBase):
    '''
    Tests tfplot.contrib module.
    '''
    def test_contrib_module(self):
        print("")
        for name in tfplot.contrib.__all__:
            fn = tfplot.contrib.__dict__[name]
            print(" - contrib: {fn} -> module={module}".format(
                  fn=fn, module=fn.__module__))
            self.assertTrue(fn.__module__.startswith('tfplot.contrib'),
                            msg=str(fn.__module__))



if __name__ == '__main__':
    unittest.main()
