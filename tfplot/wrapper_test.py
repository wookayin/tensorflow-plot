'''Unit Test for tfplot.ops'''

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

import tfplot.figure
import tfplot.test_util as test_util


class TestWrap(test_util.TestcaseBase):
    '''
    Tests tfplot.wrap() and wrap_axesplot()
    '''

    def _check_plot_op_shape(self, op):
        '''Check if op is a uint8 Tensor of shape [?, ?, 4]'''
        cprint(" op: %s" % op, color='cyan')
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
        cprint("\n tf_plot: %s" % tf_plot, color='magenta')
        self.assertIsInstance(tf_plot, types.FunctionType)

        # TensorFlow plot_op
        plot_op = tf_plot("hello world")
        self._check_plot_op_shape(plot_op)
        self.assertEqual(plot_op.name, 'Wrapped:0')

    def test_wrap_axesplot_axes(self):
        '''Basic functionality test of tfplot.wrap_axesplot() in successful cases.'''

        # (case i) an instance of matplotlib axes
        from matplotlib.axes import Axes
        tf_scatter = tfplot.wrap_axesplot(Axes.scatter)
        cprint("\n tf_scatter: %s" % tf_scatter, color='magenta')

        plot_op = tf_scatter([1, 2, 3], [1, 4, 9])
        self._check_plot_op_shape(plot_op)
        self.assertTrue(re.match('scatter(_\d)?:0', plot_op.name))

    def test_wrap_axesplot_kwarg(self):
        '''Basic functionality test of tfplot.wrap_axesplot() in successful cases.'''

        # (case ii) any unbounded function that has 'ax=...' keyword parameter
        def fn_to_wrap(ax=None):
            ax.text(0.5, 0.5, "Hello!")
            return ax
        # TODO: _fn_to_wrap has an error

        tf_plot = tfplot.wrap_axesplot(fn_to_wrap)
        cprint("\n tf_plot: %s" % tf_plot, color='magenta')
        self.assertIsInstance(tf_plot, types.FunctionType)

        # TensorFlow plot_op
        plot_op = tf_plot("hello world")
        self._check_plot_op_shape(plot_op)
        self.assertEqual(plot_op.name, 'fn_to_wrap:0')

    def test_wrap_axesplot_invalid(self):
        '''Invalid usage of tfplot.wrap_axesplot().'''
        fig, ax = tfplot.figure.subplots()

        with self.assertRaises(ValueError):
            # it should raise a ValueError about bound method
            tf_plot = tfplot.wrap_axesplot(ax.scatter)

    def test_autowrap_call_axesplot(self):
        '''Does autowrap also work with Axes.xxxx methods?
        needs to handle binding (e.g. self) carefully! '''
        from matplotlib.axes import Axes
        tf_scatter = tfplot.autowrap(Axes.scatter, name='ScatterAutowrap')
        cprint("\n tf_scatter: %s" % tf_scatter, color='magenta')

        op = tf_scatter([1, 2, 3], [1, 4, 9])
        self._execute_plot_op(op)

    def test_autowrap_call_extrakwargs(self):
        '''when calling autowrap to wrap a seaborn plot function,
        additional kwargs (non-standard) should be applied as default arguments
        for the actual py.func invocation.'''
        tf_heatmap = tfplot.autowrap(sns.heatmap, figsize=(2, 2),
                                     tight_layout=True, cmap='jet', cbar=False,
                                     xticklabels=False, yticklabels=False)
        op = tf_heatmap(tf.constant(np.eye(5)))
        r = self._execute_plot_op(op)
        self.assertEquals(test_util.hash_image(r), '528047f739fe6dc4ba4ec1738b3a44b5bc95ecff')

    def test_autowrap_call_extrakwargs_unknownargswillerror(self):
        '''additional kwargs should be applied to the py.func,
        but unknown arguments should be trigger an error.'''
        def known_func(x, ax=None):
            assert (x is not None) and (ax is not None)
        tf_heatmap = tfplot.autowrap(known_func, figsize=(2, 2),
                                     unknown_arg_should_be_error='ERRRRRRRR')
        op = tf_heatmap(tf.constant(np.eye(5)))

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        with self.assertRaises(InvalidArgumentError):
            r = self._execute_plot_op(op)



class TestDecorator(test_util.TestcaseBase):
    """
    Usage of tfplot.wrap() and tfplot.autowrap() as decorator.
    """

    def test_wrap_simple(self):
        '''Use as decorator'''
        @tfplot.wrap
        def foo():
            fig, ax = tfplot.subplots()
            ax.plot([1, 2, 3])
            fig.tight_layout()
            return fig

        self._execute_plot_op(op=foo())

    def test_wrap_withcall(self):
        '''Use as decorator, but with function call'''
        @tfplot.wrap()
        def foo():
            fig, ax = tfplot.subplots()
            ax.plot([1, 2, 3])
            fig.tight_layout()
            return fig

        self._execute_plot_op(op=foo())

    def test_wrap_withcall_argument(self):
        '''Use as decorator, but with function call with arguments'''
        @tfplot.wrap()
        def foo(values):
            fig, ax = tfplot.subplots()
            ax.plot(values)
            fig.tight_layout()
            return fig

        op = foo(tf.convert_to_tensor([2, 2, 3, 3]))
        self._execute_plot_op(op)

    def test_wrap_autoinject_figax(self):
        """Tests whether @tfplot.autowrap work in many use cases"""
        @tfplot.autowrap
        def foo_autoinject_return_fig(fig=None, ax=None):
            # fig, ax should have been autoinjected
            assert fig and isinstance(fig, matplotlib.figure.Figure)
            assert ax and isinstance(ax, matplotlib.axes.Axes)

            ax.text(0.5, 0.5, "autoinject", ha='center')
            return fig
        self._execute_plot_op(foo_autoinject_return_fig())

        @tfplot.autowrap
        def foo_autoinject_return_ax(ax=None):
            ax.text(0.5, 0.5, "autoinject", ha='center')
            return ax
        self._execute_plot_op(foo_autoinject_return_ax())

        @tfplot.autowrap
        def foo_autoinject_return_nothing(fig=None, ax=None):
            ax.text(0.5, 0.5, "autoinject", ha='center')
        self._execute_plot_op(foo_autoinject_return_nothing())

        @tfplot.wrap
        def foo_autoinject_shouldntwork(fig=None, ax=None):
            ax.text(0.5, 0.5, "autoinject", ha='center')
        with self.assertRaises(Exception):
            self._execute_plot_op(foo_autoinject_shouldntwork())

    @unittest.skipIf(sys.version_info[0] < 3, "Python 3+")
    def test_wrap_autoinject_kwonly_py3(self):
        """Tests whether @tfplot.autowrap on functions with keyword-only argument"""

        # Python2 will raise a SyntaxError, so dynamically compile the code on runtime.
        ctx = {}
        exec('''if "this is Python2 SyntaxError workaround":

        @tfplot.autowrap
        def foo_autoinject_kwonly(*, fig, ax):
            ax.text(0.5, 0.5, "autoinject-kwonly", ha='center')
            return fig

        ctx['foo_autoinject_kwonly'] = foo_autoinject_kwonly
        ''')

        foo_autoinject_kwonly = ctx['foo_autoinject_kwonly']
        self._execute_plot_op(foo_autoinject_kwonly())  # pylint: disable=missing-kwoa


    def test_wrap_autowrap_arguments(self):
        """Tests optional arguments (gigsize, tight_layouts) of @tfplot.autowrap"""

        @tfplot.autowrap(figsize=(4, 1))
        def wrappee_figsize(fig=None):
            return fig

        im = self._execute_plot_op(wrappee_figsize())
        assert im.shape[0] * 4 == im.shape[1], str(im.shape)     # e.g. 100x400

        @tfplot.autowrap(tight_layout=True)
        def wrappee_tight(fig=None):
            return fig

        im = self._execute_plot_op(wrappee_tight())


if __name__ == '__main__':
    unittest.main()
