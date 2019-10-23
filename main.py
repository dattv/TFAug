import cv2
import tensorflow as tf
from image.data_augmentation import *
import os


def main(input_dir=None, output_dir=None):
    if os.path.isdir(input_dir) == False:
        return None

    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    file_img = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if f.endswith(".jpg")]

    img = tf.placeholder(dtype=tf.uint8, shape=[None, None, None], name="img")
    rand = tf.random_uniform(shape=[], minval=0., maxval=1., name="random")

    tf_flip_lr = tf_img_flip_lr(img)
    tf_flip_ud = tf_img_flip_ud(img)
    tf_saturation = tf_img_saturation(img)
    tf_brightness = tf_img_brightness(img)
    tf_rot90 = tf_img_rot90(img)
    tf_hue = tf_img_hue(img)
    tf_constrast = tf_img_constrast(img)

    tf_zoom = tf_img_zoom(img)
    tf_zoom2 = tf_img_zoom2(img)
    tf_rot = tf_img_rorate(img)
    tf_errasing = tf_img_erasing(img)

    tf_noise = tf_img_noise(img)
    tf_shift = tf_img_shift(img)

    tf_transform = tf_image_transform(img)
    with tf.Session() as sess:

        tf.summary.FileWriter("./graphs", sess.graph)
        i = 0
        for f in file_img:
            data = cv2.imread(f)

            flip_lr, \
            flip_ud, \
            saturation, \
            brightness, \
            rot90, \
            hue, \
            constrast, \
            rand_np, \
            zoom, \
            rot, \
            zoom2, \
            erasing, \
            noise, \
            shift, \
            transform = sess.run([tf_flip_lr,
                                  tf_flip_ud,
                                  tf_saturation,
                                  tf_brightness,
                                  tf_rot90,
                                  tf_hue,
                                  tf_constrast,
                                  rand,
                                  tf_zoom,
                                  tf_rot,
                                  tf_zoom2,
                                  tf_errasing,
                                  tf_noise,
                                  tf_shift,
                                  tf_transform], feed_dict={img: data})

            cv2.imwrite(os.path.join(output_dir, "flip_lr" + "_" + str(i) + ".jpg"), flip_lr)
            cv2.imwrite(os.path.join(output_dir, "flip_ud" + "_" + str(i) + ".jpg"), flip_ud)
            cv2.imwrite(os.path.join(output_dir, "saturation" + "_" + str(i) + ".jpg"), saturation)
            cv2.imwrite(os.path.join(output_dir, "brightness" + "_" + str(i) + ".jpg"), brightness)
            cv2.imwrite(os.path.join(output_dir, "rot90" + "_" + str(i) + ".jpg"), rot90)
            cv2.imwrite(os.path.join(output_dir, "hue" + "_" + str(i) + ".jpg"), hue)
            cv2.imwrite(os.path.join(output_dir, "constrast" + "_" + str(i) + ".jpg"), constrast)
            cv2.imwrite(os.path.join(output_dir, "zoom" + "_" + str(i) + ".jpg"), zoom)
            cv2.imwrite(os.path.join(output_dir, "rot" + "_" + str(i) + ".jpg"), rot)
            cv2.imwrite(os.path.join(output_dir, "zoomv2" + "_" + str(i) + ".jpg"), zoom2)
            cv2.imwrite(os.path.join(output_dir, "errasing" + "_" + str(i) + ".jpg"), erasing)
            cv2.imwrite(os.path.join(output_dir, "noise" + "_" + str(i) + ".jpg"), noise)
            cv2.imwrite(os.path.join(output_dir, "shift" + "_" + str(i) + ".jpg"), shift)
            cv2.imwrite(os.path.join(output_dir, "transform" + "_" + str(i) + ".jpg"), transform)
            i += 1


if __name__ == '__main__':
    input_dir = "./faces94/female/9336923"
    output_dir = "./data_aug"

    main(input_dir, output_dir)
