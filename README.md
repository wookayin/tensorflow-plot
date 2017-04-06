TensorFlow Plot
===============

A [TensorFlow][tensorflow] utility for providing matplotlib-based **plot** operations
— [TensorBoard][tensorboard] ❤️ [Matplotlib][matplotlib].

It allows us to draw **_any_** [matplotlib][matplotlib] plots or figures into images,
as a part of TensorFlow computation graph.
Especially, we can easily any plot and see the result image
as an image summary in [TensorBoard][tensorboard].

<p align="center">
<img src="./assets/tensorboard-plot-summary.png" width="70%" />
</p>

Quick Overview
--------------

Simply define a python function for plotting that takes `numpy.ndarray` values as input,
draw a plot, and return it as a `matplotlib.figure.Figure` object.
Then, the API `tfplot.plot()` will wrap this function as a TensorFlow operation,
which will produce a RGB image tensor `[height, width, 3]` containg the resulting plot.

```python
import tfplot

def figure_heatmap(heatmap, cmap='jet'):
    # draw a heatmap with a colorbar
    fig, ax = tfplot.subplots(figsize=(4, 3))
    im = ax.imshow(heatmap, cmap=cmap)
    fig.colorbar(im)
    return fig

# heatmap_tensor : a float32 Tensor of shape [16, 16], for example
plot_op = tfplot.plot(figure_heatmap, [heatmap_tensor], cmap='jet')

# Or just directly add an image summary with the plot
tfplot.summary.plot("heatmap_summary", figure_heatmap, [heatmap_tensor])
```

Please take a look at the
[the showcase][examples-showcase] or [examples directory][examples-dir]
for more examples and use cases.

Note
----

Please use **object-oriented** matplotlib APIs (e.g. `Figure`, `AxesSubplot`)
instead of [pyplot] APIs (i.e. `matplotlib.pyplot` or `plt.XXX()`)
when creating and drawing plots.
This is because [pyplot] APIs are not *thread-safe*,
while the TensorFlow plot operations are usually executed in multi-threaded manners.

For example, avoid any use of `pyplot` (or `plt`):

```python
# DON'T DO LIKE THIS !!!
def figure_heatmap(heatmap):
    fig = plt.figure()
    plt.imshow(heatmap)
    return fig
```

and do it like:

```python
def figure_heatmap(heatmap):
    fig = matplotlib.figure.Figure()   # or just `fig = tfplot.Figure()`
    ax = fig.add_subplot(1, 1, 1)      # ax: AxesSubplot
    ax.imshow(heatmap)
    return fig                         # fig: Figure
```

For example, `tfplot.subplots()` is a good replacement for `plt.subplots()`
to use inside plot functions.


[matplotlib]: http://matplotlib.org/
[tensorflow]: https://www.tensorflow.org/
[tensorboard]: https://www.tensorflow.org/get_started/summaries_and_tensorboard
[pyplot]: http://matplotlib.org/api/pyplot_api.html
[examples-dir]: https://github.com/wookayin/tensorflow-plot/blob/master/examples/
[examples-showcase]: https://github.com/wookayin/tensorflow-plot/blob/master/examples/showcases.ipynb

License
-------

[MIT License](LICENSE) © Jongwook Choi
