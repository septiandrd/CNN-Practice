import cifar_tools
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import conv_visuals

names, data, labels = \
    cifar_tools.read_data('cifar-10-batches-py')

def show_some_examples(names,data,labels) :
    plt.figure()
    rows, cols = 4,4 # rows and cols number
    random_idxs = random.sample(range(len(data)), rows*cols)
    for i in range(rows*cols) :
        plt.subplot(rows,cols,i+1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.reshape(data[j,:], (24,24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cifar_examples.png')

# show_some_examples(names,data,labels)