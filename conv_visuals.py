import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cifar_tools

W = tf.Variable(tf.random_normal([5,5,1,32]))   # define the tensor representing the random filters

def show_weights(W, filename=None) :
    plt.figure()
    rows,cols = 4,8 # define rows and cols to show 32 figures
    for i in range(np.shape(W)[3]) :
        img = W[:,:,0,i]
        plt.subplot(rows,cols,i+1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename :
        plt.savefig(filename)
    else:
        plt.show()

def show_conv_results(data, filename=None) :
    plt.figure()
    rows,cols = 4,8
    for i in range(np.shape(data)[3]) :
        img = data[0,:,:,i]
        plt.subplot(rows,cols,i+1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename :
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':

    # with tf.Session() as sess :
    #     sess.run(tf.global_variables_initializer())
    #
    #     W_val = sess.run(W)
    #     show_weights(W_val, 'step0_weights.png')

    names, data, labels = \
        cifar_tools.read_data('cifar-10-batches-py')

    # get an image from CIFAR dataset and visualize it
    raw_data = data[4, :]
    raw_img = np.reshape(raw_data, (24, 24))
    plt.figure()
    plt.imshow(raw_img, cmap='Greys_r')
    plt.savefig('input_image.png')

    # define the input tensor for 24x24 image
    x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])

    # define the filters and corresponding parameters
    b = tf.Variable(tf.random_normal([32]))
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)

    # max pooling
    k = 2
    maxpool = tf.nn.max_pool(conv_out,
                             ksize=[1,k,k,1],
                             strides=[1,k,k,1],
                             padding='SAME')

    # run the convolution on selected image
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        conv_val = sess.run(conv)
        show_conv_results(conv_val, 'step1_convs.png')
        print(np.shape(conv_val))

        conv_with_b_val = sess.run(conv_with_b)
        conv_out_val = sess.run(conv_out)
        show_conv_results(conv_out_val, 'step2_conv_outs.png')
        print(np.shape(conv_out_val))

        maxpool_val = sess.run(maxpool)
        show_conv_results(maxpool_val,'step3_maxpool.png')