import numpy as np
import matplotlib.pyplot as plt
import cifar_tools
import tensorflow as tf

def conv_layer(x,W,b) :
    conv = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    conv_with_b = tf.nn.bias_add(conv,b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out

def maxpool_layer(conv, k=2) :
    return tf.nn.max_pool(conv,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def model() :

    x_reshaped = tf.reshape(x,shape=[-1,24,24,1])

    # construct first conv layer and maxpooling
    conv_out1 = conv_layer(x_reshaped, W1,b1)
    maxpool_out1 = maxpool_layer(conv_out1)
    norm1 = tf.nn.lrn(maxpool_out1,4,bias=1.0,alpha=0.001/9.0, beta=0.75)

    # construct second layer
    conv_out2 = conv_layer(norm1,W2,b2)
    norm2 = tf.nn.lrn(conv_out2,4,bias=1.0,alpha=0.001/9.0, beta=0.75)
    maxpool_out2 = maxpool_layer(norm2)

    # construct fully connected layer
    maxpool_reshaped = tf.reshape(maxpool_out2, [-1,W3.get_shape().as_list()[0]])
    local = tf.add(tf.matmul(maxpool_reshaped,W3),b3)
    local_out = tf.nn.relu(local)

    out = tf.add(tf.matmul(local_out,W_out),b_out)
    return out

if __name__ == '__main__':

    # load the dataset
    names, data, labels = \
        cifar_tools.read_data('cifar-10-batches-py')


    

    # # define input and output placeholder
    # x = tf.placeholder(tf.float32, [None, 24 * 24])
    # y = tf.placeholder(tf.float32, [None, len(names)])
    #
    # # apply 64 convolutions of 5x5
    # W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
    # b1 = tf.Variable(tf.random_normal([64]))
    #
    # # apply 64 more colvolutions of 5x5
    # W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
    # b2 = tf.Variable(tf.random_normal([64]))
    #
    # # introduce a fully-connected layer
    # W3 = tf.Variable(tf.random_normal([6 * 6 * 64, 1024]))
    # b3 = tf.Variable(tf.random_normal([1024]))
    #
    # # define variables for fully-connected linear layer
    # W_out = tf.Variable(tf.random_normal([1024, len(names)]))
    # b_out = tf.Variable(tf.random_normal([len(names)]))
    #
    # model_op = model()
    #
    # # define classification loss function
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_op,labels=y))
    #
    # # define training op to minimize the loss function
    # train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #
    # correct_pred = tf.equal(tf.argmax(model_op,1),tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #
    # with tf.Session() as sess :
    #     sess.run(tf.global_variables_initializer())
    #     onehot_labels = tf.one_hot(labels, len(names), on_value=1.,off_value=0.,axis=-1)
    #     onehot_vals = sess.run(onehot_labels)
    #     batch_size = len(data)
    #     print('batch_size', batch_size)
    #     for j in range(0,1000) :
    #         print('EPOCH',j)
    #         for i in range(0,len(data),batch_size) :
    #             batch_data = data[i:i+batch_size,:]
    #             batch_onehot_vals = onehot_vals[i:i+batch_size,:]
    #             _,accuracy_val = sess.run([train_op,accuracy], feed_dict={x:batch_data,y:batch_onehot_vals})
    #             if i%1000 == 0 :
    #                 print(i, accuracy_val)
    #         print('DONE')

