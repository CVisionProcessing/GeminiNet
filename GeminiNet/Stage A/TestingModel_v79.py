import cv2
import numpy as np
import bisober_v79
import os
import sys
import tensorflow as tf
import time
import vgg16
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)

def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = './MSRA-B/image'
    elif dataset == 'HKU-IS':
        path = './HKU-IS/imgs'
    elif dataset == 'DUT-OMRON':
        path = './DUT-OMRON/DUT-OMRON-image'
    elif dataset == 'PASCAL-S':
        path = './PASCAL-S/pascal'
    elif dataset == 'SOD':
        path = './SOD/images'
    elif dataset == 'ECSSD':
        path = './ECSSD/images'
    elif dataset == 'DUTS':
        path = './DUTS/DUTS-TE/DUTS-TE-Image'

    imgs = os.listdir(path)

    return path, imgs


if __name__ == "__main__":

    model = bisober_v79.Model()
    model.build_model()

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    img_size = bisober_v79.img_size
    label_size = bisober_v79.label_size

    ckpt = tf.train.get_checkpoint_state('/mnt/hdd1/zhengtao/bi_sq/V79_3_Model/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    datasets = ['ECSSD','DUTS','HKU-IS','MSRA-B','PASCAL-S','DUT-OMRON','SOD']

    if not os.path.exists('Result'):
        os.mkdir('Result')

    for dataset in datasets:
        path, imgs = load_img_list(dataset)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result/' + dataset + '/NLDF'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            img_name, ext = os.path.splitext(f_img) #返回文件名和拓展名

            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))

                start_time = time.time()
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder: img,model.training: False})
#                result = sess.run(model.Prob,
#                                  feed_dict={model.input_holder: img})
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0])) #np.squeeze将输入的array转换为1维

                save_name = os.path.join(save_dir, img_name+'.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
