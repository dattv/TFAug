from __future__ import print_function
import os

import numpy
import tensorflow as tf


def tf_img_flip_lr(img):
    return tf.image.random_flip_left_right(img)


def tf_img_flip_ud(img):
    return tf.image.random_flip_up_down(img)


def tf_img_rot90(img, probability=True):
    choice_rotate90 = tf.random_uniform(shape=[], minval=0., maxval=1., name="choice_rotate90")
    return tf.cond(choice_rotate90 < 0.5, lambda: img, lambda: tf.image.rot90(img))


def tf_img_hue(img, max_delta=0.5):
    return tf.image.random_hue(img, max_delta)


def tf_img_saturation(img, min=0.5, max=1.5):
    return tf.image.random_saturation(img, min, max)


def tf_img_brightness(img, max_delta=0.3):
    return tf.image.random_brightness(img, max_delta)


def tf_img_constrast(img, min=0.5, max=1.5):
    return tf.image.random_contrast(img, min, max)


def tf_img_zoom(img, percent=0.75):
    shape = tf.shape(img)
    if len(shape.shape) == 4:
        height = shape[1]
        width = shape[2]
    else:
        height = shape[0]
        width = shape[1]

    choice_zoom = tf.random_uniform(shape=[], minval=0., maxval=1., name="choice_zoom")

    resized_height, resized_width = tf.cond(choice_zoom < 0.5,
                                            lambda: (height, width),
                                            lambda: (tf.cast(tf.multiply(tf.cast(height, tf.float32), percent),
                                                             tf.int32),
                                                     tf.cast(tf.multiply(tf.cast(width, tf.float32), percent),
                                                             tf.int32)))

    crop = tf.random_crop(img, [resized_height, resized_width, 3])
    return tf.image.resize(crop, [height, width])


def tf_img_zoom2(img):
    shape = tf.shape(img)
    if len(shape.shape) == 4:
        height = shape[1]
        width = shape[2]
    else:
        height = shape[0]
        width = shape[1]

    choice_zoom2 = tf.random_uniform(shape=[], minval=0., maxval=1., name="choice_zoom2")
    percent2 = tf.random_uniform(dtype=tf.float32, shape=[], minval=0.3, maxval=0.6, name="percent2")
    resized_height, resized_width = tf.cond(choice_zoom2 < 0.5,
                                            lambda: (height, width),
                                            lambda: (tf.cast(tf.multiply(tf.cast(height, tf.float32), percent2),
                                                             tf.int32),
                                                     tf.cast(tf.multiply(tf.cast(width, tf.float32), percent2),
                                                             tf.int32)))
    zoom_in = tf.image.resize(img, [resized_height, resized_width])
    return tf.image.resize(zoom_in, [height, width])


def tf_img_rorate(img, max=numpy.sqrt(2.) / 2.):
    shape = tf.shape(img)
    if len(shape.shape) == 4:
        height = shape[1]
        width = shape[2]
    else:
        height = shape[0]
        width = shape[1]

    random_angles = tf.random.uniform(shape=[], minval=-numpy.pi / 4, maxval=numpy.pi / 4)

    rotated_images = tf.contrib.image.transform(
        img,
        tf.contrib.image.angles_to_projective_transforms(
            random_angles, tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        ))

    return rotated_images


def random_erase_np(img, sl=0.02, sh=0.4, r1=0.3):
    temp_img = img
    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    area = width * height
    for attempt in range(100):
        target_area = numpy.random.uniform(sl, sh) * area
        aspect_ratio = numpy.random.uniform(r1, 1 / r1)

        h = int(numpy.round(numpy.sqrt(target_area * aspect_ratio)))
        w = int(numpy.round(numpy.sqrt(target_area / aspect_ratio)))

        if w < width and h < height:
            x1 = numpy.random.randint(0, height - h)
            y1 = numpy.random.randint(0, width - w)
            temp_img[x1:x1 + h, y1:y1 + w, :] = numpy.random.uniform(0, 255, (h, w, channel))
            return temp_img

    return temp_img


def tf_img_erasing(img):
    temp_img = tf.identity(img, name="temp_img")
    probability = tf.random_uniform(shape=[], minval=0., maxval=1., name="probability")
    result = tf.cond(probability < 0.5,
                     lambda: img,
                     lambda: tf.py_func(random_erase_np, [temp_img], tf.uint8, False))

    return result


def tf_img_noise(img):
    shape = tf.shape(img)
    if len(shape.shape) == 4:
        height = shape[1]
        width = shape[2]
    else:
        height = shape[0]
        width = shape[1]

    noise = tf.random_uniform(shape=[height, width, 3], minval=-50, maxval=50, dtype=tf.int32)
    choice_noise = tf.random_uniform(shape=[], minval=0., maxval=1., name="choice_noise")

    img = tf.cond(choice_noise < 0.5, lambda: tf.cast(img, tf.int32), lambda: tf.cast(img, tf.int32) + noise)
    return tf.cast(tf.clip_by_value(img, clip_value_min=0, clip_value_max=255), tf.uint8)


def tf_img_shift(img):
    shape = tf.shape(img)
    if len(shape.shape) == 4:
        height = shape[1]
        width = shape[2]
        num_channel = shape[3]
    else:
        height = shape[0]
        width = shape[1]
        num_channel = shape[2]

    fluctuation_w = tf.random_uniform(shape=[], minval=1., maxval=1.4)
    fluctuation_h = tf.random_uniform(shape=[], minval=1., maxval=1.4)

    height_fluctuation = tf.cast(tf.cast(height, tf.float32) * fluctuation_h, tf.int32)
    width_fluctuation = tf.cast(tf.cast(width, tf.float32) * fluctuation_w, tf.int32)
    new_img = tf.image.resize_image_with_crop_or_pad(img, height_fluctuation, width_fluctuation)
    return tf.image.random_crop(new_img, [height, width, num_channel])


def tf_image_transform(image):
    ran = tf.random_uniform([])
    x = tf.random_uniform([], minval=-0.3, maxval=0.3)
    x_com = tf.random_uniform([], minval=1 - x - 0.1, maxval=1 - x + 0.1)

    y = tf.random_uniform([], minval=-0.3, maxval=0.3)
    y_com = tf.random_uniform([], minval=1 - y - 0.1, maxval=1 - y + 0.1)

    transforms = [x_com, x, 0, y, y_com, 0, 0.00, 0]

    ran = tf.random_uniform([])
    image = tf.cond(ran < 0.5,
                    lambda: tf.contrib.image.transform(image, transforms, interpolation='NEAREST', name=None),
                    lambda: tf.contrib.image.transform(image, transforms, interpolation='BILINEAR', name=None))
    return image
