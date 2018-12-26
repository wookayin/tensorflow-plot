from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

import hashlib
from imgcat import imgcat
from termcolor import cprint


def hash_image(img):
    if isinstance(img, bytes):
        return hashlib.sha1(img).hexdigest()
    else:
        return hashlib.sha1(img.tobytes()).hexdigest()


class TestcaseBase(tf.test.TestCase):

    def _execute_plot_op(self, op, print_image=True, feed_dict={}):
        '''
        Execute the op, and get the result (e.g. ndarray) under a test session
        '''
        with self.test_session() as sess:
            cprint("\n >>> " + str(op), color='cyan')
            self.assertIsInstance(op, tf.Tensor)
            ret = sess.run(op, feed_dict=feed_dict)
            if print_image and sys.platform == 'darwin':
                print(' ', end='')
                sys.stdout.flush()
                imgcat(ret)
                print('SHA1: ' + hash_image(ret))
            return ret
