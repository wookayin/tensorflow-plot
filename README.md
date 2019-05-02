TensorFlow Plot (tfplot)
========================

[![pypi](https://img.shields.io/pypi/v/tensorflow-plot.svg?maxAge=86400)][pypi_tfplot]
[![Documentation Status](https://readthedocs.org/projects/tensorflow-plot/badge/?version=latest)][documentation]
[![Build Status](https://travis-ci.org/wookayin/tensorflow-plot.svg?branch=master)](https://travis-ci.org/wookayin/tensorflow-plot)

A [TensorFlow][tensorflow] utility for providing matplotlib-based **plot** operations
— [TensorBoard][tensorboard] ❤️ [Matplotlib][matplotlib].

<p align="center">
<i> 🚧 Under Construction —  API might change!</i>
</p>

It allows us to draw **_any_** [matplotlib][matplotlib] plots or figures into images,
as a part of TensorFlow computation graph.
Especially, we can easily any plot and see the result image
as an image summary in [TensorBoard][tensorboard].

<p align="center">
<img src="./assets/tensorboard-plot-summary.png" width="70%" />
</p>


Quick Overview
--------------

There are two main ways of using `tfplot`: (i) Use as TF op, and (ii) Manually add summary protos.

### Usage: Decorator

You can directly declare a Tensor factory by using [`tfplot.autowrap`][tfplot-autowrap] as a decorator.
In the body of the wrapped function you can add any logic for drawing plots. Example:

```python
@tfplot.autowrap(figsize=(2, 2))
def plot_scatter(x: np.ndarray, y: np.ndarray, *, ax, color='red'):
    ax.scatter(x, y, color=color)

x = tf.constant([1, 2, 3], dtype=tf.float32)     # tf.Tensor
y = tf.constant([1, 4, 9], dtype=tf.float32)     # tf.Tensor
plot_op = plot_scatter(x, y)                     # tf.Tensor shape=(?, ?, 4) dtype=uint8
```


### Usage: Wrap as TF ops

We can [wrap][tfplot-autowrap] **any** pure python function for plotting as a Tensorflow op, such as:

- (i) A python function that creates and return a matplotlib `Figure` (see below)
- (ii) A python function that has `fig` or `ax` keyword parameters (will be auto-injected);
  e.g. [`seaborn.heatmap`](http://seaborn.pydata.org/generated/seaborn.heatmap.html)
- (iii) A method instance of [matplotlib `Axes`](https://matplotlib.org/api/axes_api.html);
  e.g. [`Axes.scatter`](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter)

Example of (i): You can define a python function that takes `numpy.ndarray` values as input (as an argument of Tensor input),
and draw a plot as a return value of `matplotlib.figure.Figure`.
The resulting TensorFlow plot op will be a RGBA image tensor of shape `[height, width, 4]` containing the resulting plot.


```python
def figure_heatmap(heatmap, cmap='jet'):
    # draw a heatmap with a colorbar
    fig, ax = tfplot.subplots(figsize=(4, 3))       # DON'T USE plt.subplots() !!!!
    im = ax.imshow(heatmap, cmap=cmap)
    fig.colorbar(im)
    return fig

heatmap_tensor = ...   # tf.Tensor shape=(16, 16) dtype=float32

# (a) wrap function as a Tensor factory
plot_op = tfplot.autowrap(figure_heatmap)(heatmap_tensor)      # tf.Tensor shape=(?, ?, 4) dtype=uint8

# (b) direct invocation similar to tf.py_func
plot_op = tfplot.plot(figure_heatmap, [heatmap_tensor], cmap='jet')

# (c) or just directly add an image summary with the plot
tfplot.summary.plot("heatmap_summary", figure_heatmap, [heatmap_tensor])
```

Example of (ii):

```python tfplot
import tfplot
import seaborn.apionly as sns

tf_heatmap = tfplot.autowrap(sns.heatmap, figsize=(4, 4), batch=True)   # function: Tensor -> Tensor
plot_op = tf_heatmap(attention_maps)   # tf.Tensor shape=(?, 400, 400, 4) dtype=uint8
tf.summary.image("attention_maps", plot_op)
```

Please take a look at the [the showcase][examples-showcase] or [examples directory][examples-dir] for more examples and use cases.

[The full documentation][documentation] including API docs can be found at [readthedocs][documentation].


### Usage: Manually add summary protos

```python
import tensorboard as tb
fig, ax = ...

# Get RGB image manually or by executing plot ops.
embedding_plot = sess.run(plot_op)                 # ndarray [H, W, 3] uint8
embedding_plot = tfplot.figure_to_array(fig)       # ndarray [H, W, 3] uint8

summary_pb = tb.summary.image_pb('plot_embedding', [embedding_plot])
summary_writer.write_add_summary(summary_pb, global_step=global_step)
```


Installation
------------

```
pip install tensorflow-plot
```

To grab the latest development version:

```
pip install git+https://github.com/wookayin/tensorflow-plot.git@master
```

Note
----

### Some comments on Speed

* Matplotlib operations can be **very** slow as Matplotlib runs in python rather than native code,
so please watch out for runtime speed.
There is still a room for improvement, which will be addressed in the near future.

* Moreover, it might be also a good idea to draw plots from the main code (rather than having a TF op) and add them as image summaries.
Please use this library at your best discernment.

### Thread-safety issue

Please use **object-oriented** matplotlib APIs (e.g. `Figure`, `AxesSubplot`)
instead of [pyplot] APIs (i.e. `matplotlib.pyplot` or `plt.XXX()`)
when creating and drawing plots.
This is because [pyplot] APIs are not *thread-safe*,
while the TensorFlow plot operations are usually executed in multi-threaded manners.

For example, avoid any use of `pyplot` (or `plt`):

```python
# DON'T DO LIKE THIS !!!
def figure_heatmap(heatmap):
    fig = plt.figure()                 # <--- NO!
    plt.imshow(heatmap)
    return fig
```

and do it like:

```python
def figure_heatmap(heatmap):
    fig = matplotlib.figure.Figure()   # or just `fig = tfplot.Figure()`
    ax = fig.add_subplot(1, 1, 1)      # ax: AxesSubplot
    # or, just `fig, ax = tfplot.subplots()`
    ax.imshow(heatmap)
    return fig                         # fig: Figure
```

For example, `tfplot.subplots()` is a good replacement for `plt.subplots()`
to use inside plot functions.
Alternatively, you can just take advantage of automatic injection of `fig` and/or `ax`.


[pypi_tfplot]: https://pypi.python.org/pypi/tensorflow-plot
[matplotlib]: http://matplotlib.org/
[tensorflow]: https://www.tensorflow.org/
[tensorboard]: https://www.tensorflow.org/get_started/summaries_and_tensorboard
[pyplot]: http://matplotlib.org/api/pyplot_api.html
[examples-dir]: https://github.com/wookayin/tensorflow-plot/blob/master/examples/
[examples-showcase]: https://github.com/wookayin/tensorflow-plot/blob/master/examples/showcases.ipynb
[documentation]: http://tensorflow-plot.readthedocs.io/en/latest/

[tfplot-autowrap]: https://tensorflow-plot.readthedocs.io/en/latest/api/tfplot.html#tfplot.autowrap


### TensorFlow compatibility

Currently, `tfplot` is compatible with TensorFlow 1.x series.
Support for eager execution and TF 2.0 will be coming soon!


License
-------

[MIT License](LICENSE) © Jongwook Choi
