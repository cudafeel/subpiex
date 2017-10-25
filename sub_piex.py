import tensorflow as tf
import os
from PIL import Image
import numpy as np
import string
cwd = 'G:\game\python\\'
classes = [1]
writer = tf.python_io.TFRecordWriter("test.tfrecords")
for i in range(100):
    img_path_h = cwd+'H\\'+str(i)+'_h.jpg'
    img_path_l = cwd+'L\\'+str(i)+'_l.jpg'
    img_h = Image.open(img_path_h)
    img_h = img_h.resize((28, 28))
    img_l = Image.open(img_path_l)
    img_l = img_l.resize((14, 14))
    img_raw_h = img_h.tobytes()
    img_raw_l = img_l.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_l])),
        "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_h]))
    }))
    writer.write(example.SerializeToString())
writer.close()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=5000) #创建一个队列
    reader = tf.TFRecordReader()
    n, serialized_exmaple = reader.read(filename_queue) #返回文件名和文件
    features = tf.parse_single_example(serialized_exmaple,features={
        "label":tf.FixedLenFeature([], tf.string),
        "img_raw":tf.FixedLenFeature([], tf.string),
    }) #解析tfrecord文件
    img_h = tf.decode_raw(features['img_raw'],tf.uint8)
    img_h_l = tf.reshape(img_h, [28,28])
    img_h = tf.cast(img_h_l,tf.float32)* (1. / 255) - 0.5

    img_l = tf.decode_raw(features['label'], tf.uint8)
    img_l_l = tf.reshape(img_l, [14, 14])
    img_l = tf.cast(img_l_l, tf.float32) * (1. / 255) - 0.5
    #print(img_l)
    return img_l,img_h,img_h_l,img_l_l   #run 之后返回的是list
'''
for serialized_example in tf.python_io.tf_record_iterator("test.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    print(image,label)
'''


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])  #更新列表
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):  #with之后的语句在 ema_apply_op执行之后进行
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, num_or_size_splits=a, axis=1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, b, a*r, r
    X = tf.split(X, num_or_size_splits=b, axis=1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], axis=3)
    else:
        X = _phase_shift(X, r)
    return X

def CNNlayer(image_batch):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('w1',[5, 5, 1, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))   #初始化，截断正态分布
        biases = tf.get_variable('b1', [64], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(image_batch,kernel, [1, 1, 1, 1], padding='SAME',name='conv')
        conv1 = tf.nn.bias_add(conv1,bias=biases,name='bias_add')
        conv1 = tf.nn.tanh(conv1,name='layer1')
    with tf.variable_scope('conv2') as scope:
        kernel2 = tf.get_variable('w2', [3, 3, 64, 32],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))  # 初始化，截断正态分布
        biases2 = tf.get_variable('b2', [32], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(conv1, kernel2, [1, 1, 1, 1], padding='SAME',name='conv')
        conv2 = tf.nn.bias_add(conv2, bias=biases2,name='bias_add')
        conv2 = tf.nn.tanh(conv2, name='layer2')
    with tf.variable_scope('conv3') as scope:
        kernel3 = tf.get_variable('w3', [3, 3, 32, 4],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))  # 初始化，截断正态分布
        biases3 = tf.get_variable('b3', [4], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(conv2, kernel3, [1, 1, 1, 1], padding='SAME',name='conv')
        conv3 = tf.nn.bias_add(conv3, bias=biases3,name='bias_add')
        conv3 = tf.nn.tanh(conv3, name='layer3')
        return conv3

def model(image_batch):
    CNN_out = CNNlayer(image_batch)
    out = PS(CNN_out, 2)
    return out

def loss(img,label):
    loss_temp = tf.square(tf.subtract(img,label),name='loss')
    loss_temp = tf.reduce_mean(loss_temp)
    tf.summary.scalar('loss', loss_temp)
    return loss_temp

def run_train(learn_rating = 0.01, batch_size = 64,epoch = 10000):
    X = tf.placeholder(tf.float32,[None,14,14])
    Y = tf.placeholder(tf.float32,[None,28,28])
    img, label,oringal,suboringal = read_and_decode("test.tfrecords")
    img_batch, label_batch, oringal_batch,suboringal_batch = tf.train.shuffle_batch([img, label,oringal,suboringal],  # 训练batch
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)
    X_image = tf.reshape(X,[-1,14,14,1])
    Y_image = tf.reshape(Y,[-1,28,28,1])
    x = CNNlayer(X_image)
    x_ps = PS(x, 2)
    x_pre = tf.cast((x_ps+0.5)*255, tf.uint8)
    suqr_loss = loss(x_ps,Y_image)
    train_op = tf.train.AdamOptimizer(learn_rating).minimize(suqr_loss)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    coord = tf.train.Coordinator()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter("E:\Python\subpiex1", sess.graph)
        threads = tf.train.start_queue_runners(coord=coord)
        for k in range(epoch):
            imgout, labelout,oringalout,suboringalout = sess.run([img_batch, label_batch,oringal_batch, suboringal_batch])  # 这里直接返回的nparray
            suboringalout = np.array(suboringalout)
            #print(suboringalout.shape)
            _, summ = sess.run([train_op,merged],feed_dict={X:imgout,Y:labelout})
            writer.add_summary(summ,k)
            #labelout_r = (np.array(labelout)+0.5)*255
           # labelout_r = np.array(labelout_r)
            #labelout_r.astype(np.uint8)

            if k%100 == 0:
                print(sess.run(suqr_loss,feed_dict={X:imgout,Y:labelout}),"迭代次数：",k)
            if k == 4100:
                x_ps_out = sess.run(x_pre, feed_dict={X:imgout, Y:labelout})
                x_ps_out_array = np.array(x_ps_out)
                x_ps_out_resize = x_ps_out_array.reshape([-1,28,28])
                for j in range(batch_size):
                    img_rebuild = x_ps_out_resize[j, :, :]
                    labelout_re = oringalout[j,:,:]
                    subout_re = suboringalout[j,:,:]
                    #print(subout_re.shape)
                    label_out = Image.fromarray(labelout_re,'L')
                    img_out = Image.fromarray(img_rebuild,'L')
                    sub_out = Image.fromarray(subout_re,'L')
                    sub_out.save(cwd+str(j)+'l.jpg')
                    img_out.save(cwd+str(j)+'r.jpg')
                    label_out.save(cwd + str(j) + 'h.jpg')

        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    run_train()
'''
img, label = read_and_decode("test.tfrecords")
img_batch,label_batch = tf.train.shuffle_batch([img,label],                #训练batch
                                                batch_size=2, capacity=20,
                                                min_after_dequeue=10)

init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1):
        imgout,labelout = sess.run([img_batch,label_batch])  #这里直接返回的nparray
        print(imgout.shape)
        imgout1 = np.array(imgout)
        labelout1 = np.array(labelout)
        #imgout2 = np.squeeze(imgout1,axis=0)
        for j in range(2):
            imgh = imgout1[j]
            imgl = labelout1[j]
       #     print(img_batch.shape)
            img_batch_h = Image.fromarray(imgh, 'L')
            label_batch_l = Image.fromarray(imgl, 'L')
            img_batch_h.save(cwd+str(j)+'h.jpg')
            label_batch_l.save(cwd + str(j) + 'l.jpg')
    coord.request_stop()
    coord.join(threads)
'''