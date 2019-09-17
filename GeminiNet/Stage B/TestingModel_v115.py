import cv2
import numpy as np
import bisober_v115
import os
import sys
import tensorflow as tf
import time
import vgg16
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
gpu_options = tf.GPUOptions(allow_growth=True)

def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = './MSRA-B/image'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/MSRA-B/NLDF'
    elif dataset == 'HKU-IS':
        path = './HKU-IS/imgs'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/HKU-IS/NLDF'
    elif dataset == 'DUT-OMRON':
        path = './DUT-OMRON/DUT-OMRON-gt-pixelwise'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/DUT-OMRON/NLDF'
    elif dataset == 'PASCAL-S':
        path = './PASCAL-S/pascal_gt'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/PASCAL-S/NLDF'
    elif dataset == 'SOD':
        path = './SOD/GT'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/SOD/NLDF'
    elif dataset == 'ECSSD':
        path = './ECSSD/images'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/ECSSD/NLDF'
    elif dataset == 'DUTS':
        path = './DUTS/DUTS-TE/DUTS-TE-Mask'
        pr_path = '/mnt/hdd1/zhengtao/all_result/0.940/DUTS/NLDF'

    imgs = os.listdir(path)

    return path, imgs,pr_path


if __name__ == "__main__":

    model = bisober_v115.Model()
    model.build_model()

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    img_size = bisober_v115.img_size
    label_size = bisober_v115.label_size

    ckpt = tf.train.get_checkpoint_state('/mnt/hdd1/zhengtao/bi_sq/V115_1_Model/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    datasets = ['ECSSD','DUTS','HKU-IS','PASCAL-S','DUT-OMRON','SOD']
#    datasets = ['SOD']

    if not os.path.exists('0.940_v115'):
        os.mkdir('0.940_v115')

    for dataset in datasets:
        path, imgs, pr_path = load_img_list(dataset)

        save_dir = '0.940_v115/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = '0.940_v115/' + dataset + '/NLDF'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f_img in imgs:
            img = cv2.imread(os.path.join(path, f_img)).astype(np.float32)
            priors = cv2.imread(os.path.join(pr_path, f_img[:-4]+ '.png'))[:, :, 0].astype(np.float32)
            img_name, ext = os.path.splitext(f_img) #返回文件名和拓展名

            if img is not None and priors is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))
                priors = priors / 255
                priors = cv2.resize(priors, (img_size, img_size))
                priors = priors.reshape((1, img_size, img_size, 1))

                start_time = time.time()
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder: img, model.label_noise:priors})
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0])) #np.squeeze将输入的array转换为1维

                save_name = os.path.join(save_dir, img_name+'.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
