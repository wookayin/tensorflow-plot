'''Unit Test for tfplot.summary'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import types
import sys
import os
import re

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # filter out INFO and WARN logs
tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib
matplotlib.rcParams['figure.figsize'] = (2.5, 2.5)

from imgcat import imgcat
from termcolor import cprint
import seaborn as sns
import numpy as np

import tfplot.summary
import tfplot.test_util as test_util


Summary = tf.Summary


class TestSummary(test_util.TestcaseBase):
    '''
    Tests tfplot.summary
    '''

    def _execute_summary_op(self, op, feed_dict={}):
        '''
        Execute the summary op, and parse the result into Summary proto object.
        '''
        with self.test_session() as sess:
            cprint("\n >>> " + str(op), color='magenta')
            self.assertIsInstance(op, tf.Tensor)
            self.assertTrue(op.dtype, tf.string)
            ret = sess.run(op, feed_dict=feed_dict)

            # check ret is a byte
            self.assertIsInstance(ret, bytes)
            summary = tf.Summary()
            summary.ParseFromString(ret)
            return summary


    def test_summary_plot(self):
        '''tests tfplot.summary.plot'''

        def test_figure(text):
            fig, ax = tfplot.subplots(figsize=(3, 2))
            ax.text(0.5, 0.5, text, ha='center')
            return fig

        summary_op = tfplot.summary.plot("text/hello", test_figure, ["Hello Summary"])
        s = self._execute_summary_op(summary_op)

        # pylint: disable=no-member
        self.assertTrue(s.value[0].tag.startswith('text/hello'))
        self.assertEquals(s.value[0].image.width, 300)    # default dpi = 100
        self.assertEquals(s.value[0].image.height, 200)   # default dpi = 100
        png = s.value[0].image.encoded_image_string
        # pylint: enable=no-member

        if sys.platform == 'darwin':
            imgcat(png)
        self.assertEquals(test_util.hash_image(png), 'dbb47a3281626678894084fa58066f69a2570df4')


    def test_summary_wrap_batch(self):
        '''tests tfplot.summary.wrap'''

        summary_heatmap = tfplot.summary.wrap(sns.heatmap, figsize=(2, 2), cmap='jet',
                                              batch=True)

        batch_size = 3
        summary_op = summary_heatmap("heatmap_1",
                                     tf.constant(np.random.RandomState(42).normal(size=[batch_size, 4, 4])),
                                     max_outputs=2,
                                     )
        s = self._execute_summary_op(summary_op)

        self.assertEquals(len(s.value), 2)
        self.assertEquals(s.value[0].tag, ('heatmap_1/image/0'))
        self.assertEquals(s.value[1].tag, ('heatmap_1/image/1'))

        if sys.platform == 'darwin':
            imgcat(s.value[0].image.encoded_image_string)
            imgcat(s.value[1].image.encoded_image_string)


    def test_summary_wrap_nobatch(self):
        '''tests tfplot.summary.wrap'''

        summary_heatmap = tfplot.summary.wrap(sns.heatmap, figsize=(2, 2), cmap='jet',
                                              batch=False)

        summary_op = summary_heatmap("heatmap_1",
                                     tf.constant(np.random.RandomState(42).normal(size=[4, 4])),
                                     )
        s = self._execute_summary_op(summary_op)

        self.assertEquals(len(s.value), 1)
        self.assertEquals(s.value[0].tag, ('heatmap_1/image/0'))

        if sys.platform == 'darwin':
            imgcat(s.value[0].image.encoded_image_string)



if __name__ == '__main__':
    unittest.main()
