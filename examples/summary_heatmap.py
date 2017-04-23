"""
An example application that uses `tfplot` to create plot summaries and
add them into TensorBoard as image summaries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tfplot
import numpy as np
import os.path
import scipy.misc
import scipy.ndimage
import skimage.data

import seaborn.apionly as sns

try:
    import better_exceptions
except ImportError:
    pass


def make_temp_directory():
    t = './train_dir'
    if not os.path.exists(t):
        os.makedirs(t)
    return t


# an example figure function for drawing heatmap
# with overlaid background images
def heatmap_overlay(data, overlay_image=None, cmap='jet',
                    cbar=False, show_axis=False, alpha=0.5, **kwargs):
    fig, ax = tfplot.subplots(figsize=(5, 4) if cbar else (4, 4))
    fig.subplots_adjust(0, 0, 1, 1)  # use tight layout (no margins)
    ax.axis('off')

    if overlay_image is None: alpha = 1.0
    sns.heatmap(data, ax=ax, alpha=alpha, cmap=cmap, cbar=cbar, **kwargs)

    if overlay_image is not None:
        h, w = data.shape[0]
        ax.imshow(overlay_image, extent=[0, h, 0, w])

    if show_axis:
        ax.axis('on')
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    return fig


def main():
    # (1) load image
    image_0 = skimage.data.chelsea()
    image_1 = skimage.data.coffee()

    print ('image_0 : %s' % str(image_0.shape))
    print ('image_1 : %s' % str(image_1.shape))

    image_0 = tf.image.resize_image_with_crop_or_pad(image_0, 320, 320)
    image_1 = tf.image.resize_image_with_crop_or_pad(image_1, 320, 320)
    image_batch = tf.stack([image_0, image_1], name='image_batch')
    print ('image_batch : %s' % image_batch)
    tf.summary.image("image/batch", image_batch)


    # (2) generate fake attention (in a different scale)
    attention = np.zeros([2, 16, 16], dtype=np.float32)
    attention[(0, 12, 8)] = 1.0
    attention[(0, 10, 9)] = 1.0
    attention[1, :, :] = 1. / 256
    attention[1, 0, 0] = 0.1
    attention[1, 7, 9] = 0.2
    attention[0] = scipy.ndimage.filters.gaussian_filter(attention[0], sigma=1.5)

    attention_heatmap = tf.convert_to_tensor(attention)
    tf.summary.image("attention/image_summary",
                     tf.expand_dims(attention_heatmap, 3)  # make 4-d
                     )


    # (3) attention & heatmap plots
    # build a summary factory which exposes a similar interface to tf.summary.xxx()
    summary_heatmap = tfplot.summary.wrap(heatmap_overlay, batch=True)

    summary_heatmap("attention/heatmap", attention_heatmap)
    summary_heatmap("attention/heatmap_cbar", attention_heatmap,
                    cbar=True)
    summary_heatmap("attention/heatmap_axis", attention_heatmap,
                    show_axis=True)
    summary_heatmap("attention/heatmap_cmap", attention_heatmap,
                    cbar=True, cmap='jet')
    summary_heatmap("image/heatmap_overlay", attention_heatmap, image_batch,
                    cbar=True, show_axis=True, cmap='jet')
    summary_heatmap("image/heatmap_overlay_bg", attention_heatmap, image_batch,
                    alpha=0.7, cmap='gray')

    summary_op = tf.summary.merge_all()

    # -------------------------------------------------
    # execute it

    session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    tmp_dir = make_temp_directory()

    summary = session.run(summary_op)

    summary_writer = tf.summary.FileWriter(tmp_dir)
    summary_writer.add_summary(summary)

    print ("Summary written at %s" % tmp_dir)
    print ("To open tensorboard: $ tensorboard --logdir %s" % tmp_dir)

if __name__ == '__main__':
    main()
