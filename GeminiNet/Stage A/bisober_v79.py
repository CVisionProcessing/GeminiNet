import tensorflow as tf
import vgg16
import cv2
import numpy as np
from tflearn.layers.conv import global_avg_pool
from tensorflow.python.ops import array_ops

img_size = 256
label_size = img_size
fea_dim = 128
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Fully_connected(x, units=2, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [label_size*label_size, 2])
        self.training=tf.placeholder(tf.bool, shape=[])
    
    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)
            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = tf.nn.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation
            return scale
    def residual_block(self, input_x, in_dim, out_dim, training, layer_name):
        with tf.variable_scope(layer_name) as scope:
            output = tf.nn.relu(self.batch_norm(self.Conv_2d(input_x, [1, 1, in_dim, int(in_dim/2)], 0.01, name='conv_1'),training=training))
            output = tf.nn.relu(self.batch_norm(self.Conv_2d(output, [3, 3, int(in_dim/2), int(in_dim/2)], 0.01, name='conv_2'),training=training))
            output = self.batch_norm(self.Conv_2d(output, [1, 1, int(in_dim/2), out_dim], 0.01, name='conv_3'),training=training)
            output = output+input_x
            output = tf.nn.relu(output)
            return output
    def residual_layer(self, input_x, in_dim, x_dim, training, layer_name):
        with tf.variable_scope(layer_name) as scope:
            trunk = self.residual_block(input_x,in_dim, in_dim,self.training,"trunk_1")
            trunk = self.residual_block(trunk,in_dim, in_dim,self.training,"trunk_2")
            residual = tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            residual = self.residual_block(residual,in_dim,in_dim,self.training,"res_1")
            skip = self.residual_block(residual,in_dim,in_dim,self.training,"skip")
            residual = tf.nn.max_pool(residual, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            residual = self.residual_block(residual,in_dim,in_dim,self.training,"res_2")
            residual = self.residual_block(residual,in_dim,in_dim,self.training,"res_3")
            residual = tf.image.resize_images(residual,[int(x_dim/2),int(x_dim/2)])
            residual = residual+skip
            residual = self.residual_block(residual,in_dim,in_dim,self.training,"res_4")
            residual = tf.image.resize_images(residual,[x_dim,x_dim])
            residual = tf.nn.relu(self.batch_norm(self.Conv_2d(residual, [1, 1, in_dim, in_dim], 0.01, name='res_5'),training=training))
            residual = self.Conv_2d(residual, [1, 1, in_dim, in_dim], 0.01, name='res_6')
            residual = tf.nn.sigmoid(residual)
            output = (1+residual)*trunk
            return output

    def build_model(self):
        # gbd
        vgg = self.vgg
        vgg.build(self.input_holder)

        #注意，他这里的bi-direction和论文里面是相反的
        conv5_dilation = self.dilation(vgg.conv5_3, 512, 128, 'conv5') #16x16
        conv5_dilation = self.dilation(conv5_dilation, 512, 128, 'conv5_doub')
        conv4_dilation = self.dilation(vgg.conv4_3, 512, 128, 'conv4') #32x32
        conv4_dilation = self.dilation(conv4_dilation, 512, 128, 'conv4_doub')
        conv3_dilation = self.dilation(vgg.conv3_3, 256, 128, 'conv3') #64x64
        conv3_dilation = self.dilation(conv3_dilation, 512, 128, 'conv3_doub')
        conv2_dilation = self.dilation(vgg.conv2_2, 128, 128, 'conv2') #128x128
        conv2_dilation = self.dilation(conv2_dilation, 512, 128, 'conv2_doub')
        conv1_dilation = self.dilation(vgg.conv1_2, 64, 128, 'conv1')
        conv1_dilation = self.dilation(conv1_dilation, 512, 128, 'conv1_doub')
        with tf.variable_scope('fusion') as scope:

            att_1= self.squeeze_excitation_layer(conv1_dilation, out_dim=512, ratio=4, layer_name='squeeze_layer_1')
            att_3= self.squeeze_excitation_layer(conv2_dilation, out_dim=512, ratio=4, layer_name='squeeze_layer_2')
            att_5= self.squeeze_excitation_layer(conv3_dilation, out_dim=512, ratio=4, layer_name='squeeze_layer_3')
            att_7= self.squeeze_excitation_layer(conv4_dilation, out_dim=512, ratio=4, layer_name='squeeze_layer_4')
            att_9= self.squeeze_excitation_layer(conv5_dilation, out_dim=512, ratio=4, layer_name='squeeze_layer_5')

            temp_9=tf.image.resize_images(att_9,[32,32])
            temp_7=tf.image.resize_images(att_7,[64,64])
            temp_5=tf.image.resize_images(att_5,[128,128])
            temp_3=tf.image.resize_images(att_3,[256,256])

            h7_9=tf.nn.relu(self.Conv_2d(tf.concat([temp_9, att_7], axis=3), [3, 3, 1024, 256], 0.01, name='h7_9'))
            h5_7=tf.nn.relu(self.Conv_2d(tf.concat([temp_7, att_5], axis=3), [3, 3, 1024, 256], 0.01, name='h5_7'))
            h3_5=tf.nn.relu(self.Conv_2d(tf.concat([temp_5, att_3], axis=3), [3, 3, 1024, 256], 0.01, name='h3_5'))
            h1_3=tf.nn.relu(self.Conv_2d(tf.concat([temp_3, att_1], axis=3), [3, 3, 1024, 256], 0.01, name='h1_3'))

            prev4 = tf.nn.relu(self.Conv_2d(h7_9 , [3, 3, 256, 80], 0.01, name='prev4'))
            prev4_up = tf.image.resize_images(prev4, [64, 64])  
                
            prev3 = tf.nn.relu(self.Conv_2d(tf.concat([prev4_up, h5_7], axis=3), [3, 3, 336, 80], 0.01, name='prev3'))
            prev3_up = tf.image.resize_images(prev3, [128, 128])
            prev4_up = tf.image.resize_images(prev4, [128, 128])  
        
            prev2 = tf.nn.relu(self.Conv_2d(tf.concat([prev4_up, prev3_up, h3_5], axis=3), [3, 3, 416, 80], 0.01, name='prev2'))
            prev2_up = tf.image.resize_images(prev2, [256, 256])
            prev3_up = tf.image.resize_images(prev3, [256, 256])
            prev4_up = tf.image.resize_images(prev4, [256, 256]) 
 
            prev1 = tf.nn.relu(self.Conv_2d(tf.concat([prev4_up,prev3_up,prev2_up, h1_3], axis=3), [3, 3, 496, 80], 0.01, name='prev1'))
            
            prev4 = tf.image.resize_images(prev4, [256, 256])
            prev3 = tf.image.resize_images(prev3, [256, 256])
            prev2 = tf.image.resize_images(prev2, [256, 256])
            prev = self.fusion(tf.concat([prev1, prev2, prev3, prev4],axis=3), 320, name='prev')
            
        self.Score = tf.reshape(prev, [-1, 2])
        self.Prob = tf.nn.softmax(self.Score)
        self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score, labels=self.label_holder))
        self.correct_prediction = tf.equal(tf.argmax(self.Score, 1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def focal_loss(self,prediction_tensor, target_tensor, gamma=2):
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
        per_entry_cross_ent = - (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) - (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent)

    def fusion(self,input_,input_dim,name):
        with tf.variable_scope(name) as scope:
            a = self.Atrous_conv2d(input_, [3, 3, input_dim, 2], 1, 0.01, name = 'fusion1')
            b = self.Atrous_conv2d(input_, [3, 3, input_dim, 2], 3, 0.01, name = 'fusion3')
            c = self.Atrous_conv2d(input_, [3, 3, input_dim, 2], 5, 0.01, name = 'fusion5')
            d = self.Atrous_conv2d(input_, [3, 3, input_dim, 2], 7, 0.01, name = 'fusion7')
            e = a+b+c+d
        return e
    
    def dilation(self,input_,input_dim,output_dim,name):
        with tf.variable_scope(name) as scope:
            a = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 1, 0.01, name = "dilation1"))
            b = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 3, 0.01, name = 'dilation3'))
            c = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 5, 0.01, name = 'dilation5'))
            d = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 7, 0.01, name = 'dilation7'))
            e = tf.concat([a,b,c,d],axis = 3)
        return e
    
    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            # b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            b = tf.get_variable('b', shape=[shape[3]],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)

            return conv

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv
    def Atrous_conv2d(self,input_,shape,rate,stddev,name,padding = 'SAME'):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                            shape = shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
            atrous_conv = tf.nn.atrous_conv2d(input_,W,rate = rate,padding=padding)
            b = tf.get_variable('b', shape=[shape[3]], initializer=tf.constant_initializer(0.0))
            atrous_conv = tf.nn.bias_add(atrous_conv, b)
        return atrous_conv
    def conv_2d_str(self, input_, shape, stddev, name, stride=2, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',shape=shape,initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, W, [1, stride, stride, 1], padding=padding)
            b = tf.get_variable('b', shape=[shape[3]],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
            return conv
    def batch_norm(self,inputs, training=True):
        return tf.layers.batch_normalization(
            inputs=inputs, axis = 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=training, fused=True)
    def dense_layer(self, input_, name, concat_dim, input_dim=64, input_concat=None):
        with tf.variable_scope(name) as scope:
            conv = tf.nn.relu(
                self.batch_norm(self.Conv_2d(input_, [3, 3, input_dim, 64], 0.01, name=name+'conv1', padding='SAME'),
                                training=self.training))
            if input_concat!=None:
                concat = tf.concat([conv, input_concat], axis=3)
                output = self.Conv_2d(concat, [1, 1, concat_dim+input_dim, 64], 0.01, name=name+'conv2', padding='SAME')
            else:
                concat = tf.concat([conv, input_], axis=3)
                output = self.Conv_2d(concat, [1, 1, 64+input_dim, 64], 0.01, name=name+'conv2', padding='SAME')
        return output, concat
    def dense_block(self, input_, name, input_dim=32):
        with tf.variable_scope(name) as scope:
            layer1, concat1 = self.dense_layer(input_, name='layer1', concat_dim=input_dim, input_dim=input_dim)
            layer2, concat2 = self.dense_layer(layer1, name='layer2', concat_dim=input_dim+64, input_concat=concat1)
            layer3, concat3 = self.dense_layer(layer2, name='layer3', concat_dim=input_dim+128, input_concat=concat2)
            layer4, concat4 = self.dense_layer(layer3, name='layer4', concat_dim=input_dim+192, input_concat=concat3)
            output = tf.concat([layer1, layer2, layer3, layer4], axis=3)
        return output
