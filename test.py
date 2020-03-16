import numpy as np
import tensorflow as tf

labels = np.array([-1., 3., 5.])
logits = np.array([1., 1., 1.])

with tf.Session() as sess:
    print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)))
    print(tf.ones_like(labels).eval())
