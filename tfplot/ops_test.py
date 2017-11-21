'''Unit Test for tfplot.ops'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import types

import tensorflow as tf

import tfplot.figure


class TestWrap(unittest.TestCase):
    '''
    Tests tfplot.wrap() and wrap_axesplot()
    '''
    def _check_plot_op_shape(self, op):
        '''Check if op is a uint8 Tensor of shape [?, ?, 4]'''
        self.assertIsInstance(op, tf.Tensor)

        self.assertEqual(len(op.get_shape()), 3)
        self.assertTrue(op.get_shape().is_compatible_with([None, None, 4]))   # RGB-A
        self.assertEqual(op.dtype, tf.uint8)


    def test_wrap_simplefunction(self):
        '''Basic functionality test of tfplot.wrap() in successful cases.'''

        def _fn_to_wrap(message="str"):
            fig, ax = tfplot.figure.subplots()
            ax.text(0.5, 0.5, message)
            return fig

        # the function to create TensorFlow ops
        tf_plot = tfplot.wrap(_fn_to_wrap, name='Wrapped')
        print("tf_plot:", tf_plot)
        self.assertIsInstance(tf_plot, types.FunctionType)

        # TensorFlow plot_op
        plot_op = tf_plot("hello world")
        print("plot_op:", plot_op)
        self._check_plot_op_shape(plot_op)
        self.assertEqual(plot_op.name, 'Wrapped:0')


    def test_wrap_axesplot_axes(self):
        '''Basic functionality test of tfplot.wrap_axesplot() in successful cases.'''

        # (case i) an instance of matplotlib axes
        from matplotlib.axes import Axes
        tf_scatter = tfplot.wrap_axesplot(Axes.scatter)
        print("tf_scatter:", tf_scatter)

        plot_op = tf_scatter([1, 2, 3], [1, 4, 9])
        print("plot_op:", plot_op)

        self._check_plot_op_shape(plot_op)
        self.assertEqual(plot_op.name, 'scatter:0')

    def test_wrap_axesplot_kwarg(self):
        '''Basic functionality test of tfplot.wrap_axesplot() in successful cases.'''

        # (case ii) any unbounded function that has 'ax=...' keyword parameter
        def fn_to_wrap(ax=None):
            ax.text(0.5, 0.5, "Hello!")
            return ax
        # TODO: _fn_to_wrap has an error

        tf_plot = tfplot.wrap_axesplot(fn_to_wrap)
        print("tf_plot:", tf_plot)
        self.assertIsInstance(tf_plot, types.FunctionType)

        # TensorFlow plot_op
        plot_op = tf_plot("hello world")
        print("plot_op:", plot_op)
        self._check_plot_op_shape(plot_op)
        self.assertEqual(plot_op.name, 'fn_to_wrap:0')

    def test_wrap_axesplot_invalid(self):
        '''Invalid usage of tfplot.wrap_axesplot().'''
        fig, ax = tfplot.figure.subplots()

        with self.assertRaises(ValueError):
            # it should raise a ValueError about bound method
            tf_plot = tfplot.wrap_axesplot(ax.scatter)


if __name__ == '__main__':
    unittest.main()
