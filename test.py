import model
import numpy as np
import keras.backend as K
import tensorflow as tf
import cv2
import time

net = model.build(640, 320, 66, train=True)

sess = K.get_session()

writer = tf.summary.FileWriter('logs', sess.graph)
writer.close()

data_dict = np.load('./icnet_cityscapes_trainval_90k.npy').item()
for layer in net.layers:
    layer_name = layer.get_config()['name']
    try:
        kernel, bias = data_dict[layer_name]['weights'], data_dict[layer_name]['biases']
        layer.set_weights([kernel, bias])
    except:
        print layer_name
        continue

net.save_weights('weights.h5')

img = np.array([cv2.resize(cv2.imread('test_1024x2048.png', 1), (640,320))])
learning_phase = K.learning_phase()

while True:
    start_time = time.time()
    preds = sess.run(net.output, feed_dict={net.input: img, learning_phase: 0})
    print("--- %s seconds ---" % (time.time() - start_time))