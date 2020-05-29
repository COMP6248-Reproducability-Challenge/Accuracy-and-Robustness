import tensorflow as tf
import numpy as np
import time

from cifar import CIFAR, CIFARModel
from mnist import MNIST, MNISTModel

from attack import L2
from attack import Li


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack = L2(sess, model)
        attack = Li(sess, model)
        
        inputs, targets = generate_data(data, samples=10, targeted=False, start=10, inception=False)
        adv = attack.attack(inputs, targets)

        for i in range(len(adv)):
            print("Distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
