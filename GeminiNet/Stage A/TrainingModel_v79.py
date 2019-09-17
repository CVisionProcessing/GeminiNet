import cv2
import numpy as np
import bisober_v79
import vgg16
import tensorflow as tf
import os
import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
gpu_options = tf.GPUOptions(allow_growth=True)


def load_train_val_ecssd():
    files = []
    labels = []
    files_ecssd=[]
    labels_ecssd=[]

    with open('./DUTS/DUTS-TR/dut_train.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('./DUTS/DUTS-TR/DUTS-TR-Mask/%s' % line)
        files.append('./DUTS/DUTS-TR/DUTS-TR-Image/%s' % line.replace('.png', '.jpg'))

    with open('./ECSSD/ecssd.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        files_ecssd.append('./ECSSD/images/%s' % line.replace('.png', '.jpg'))
        labels_ecssd.append('./ECSSD/gt/%s' % line)

    return files, labels, files_ecssd, labels_ecssd

if __name__ == "__main__":

    model = bisober_v79.Model()
    model.build_model()

#    sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.Loss_Mean, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(1e-6)
    train_op = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('/mnt/hdd1/zhengtao/bi_sq/V79_2_Model')
    saver = tf.train.Saver(max_to_keep=32)
    saver.restore(sess, ckpt.model_checkpoint_path)

    train_list, label_list, ecssd_list, labels_ecssd = load_train_val_ecssd()

    n_epochs = 32
    img_size = bisober_v79.img_size
    label_size = bisober_v79.label_size
    os.mkdir('/mnt/hdd1/zhengtao/bi_sq/V79_3_Model')
    f = open("/mnt/hdd1/zhengtao/bi_sq/v79_3.txt", "w") 
    for i in range(n_epochs):
        whole_loss = 0.0
        whole_acc = 0.0
        count = 0
        ecssd_loss=0.0
        ecssd_acc=0.0
        ecssd_count=0
        
        for f_img, f_label in zip(train_list, label_list):
            img = cv2.imread(f_img).astype(np.float32)
            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            
            if(i%2==0):
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                label = cv2.resize(label, (label_size, label_size))
                label = label.astype(np.float32) / 255.
                label = np.stack((label, 1-label), axis=2)
                label = np.reshape(label, [-1, 2])
                img = img.reshape((1, img_size, img_size, 3))
                _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                    feed_dict={model.input_holder: img,
                                               model.label_holder: label,
                                               model.training: True})
                whole_loss += loss
                whole_acc += acc
                count = count + 1
            if(i%2==1):
                # add horizon flip image for training
                img_flip = cv2.flip(img, 1)
                label_flip = cv2.flip(label, 1)
                
                img_flip = cv2.resize(img_flip, (img_size, img_size)) - vgg16.VGG_MEAN
                label_flip = cv2.resize(label_flip, (label_size, label_size))
                label_flip = label_flip.astype(np.float32) / 255.
                img_flip = img_flip.reshape((1, img_size, img_size, 3))
                label_flip = np.stack((label_flip, 1 - label_flip), axis=2)
                label_flip = np.reshape(label_flip, [-1, 2])
                _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                    feed_dict={model.input_holder: img_flip,
                                               model.label_holder: label_flip,
                                               model.training: True})
                whole_loss += loss
                whole_acc += acc
                count = count + 1
            # if(i%4==1):
            #     # add vertical flip image for training
            #     img_flip = cv2.flip(img, 0)
            #     label_flip = cv2.flip(label, 0)
                
            #     img_flip = cv2.resize(img_flip, (img_size, img_size)) - vgg16.VGG_MEAN
            #     label_flip = cv2.resize(label_flip, (label_size, label_size))
            #     label_flip = label_flip.astype(np.float32) / 255.
            #     img_flip = img_flip.reshape((1, img_size, img_size, 3))
            #     label_flip = np.stack((label_flip, 1 - label_flip), axis=2)
            #     label_flip = np.reshape(label_flip, [-1, 2])
            #     _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
            #                         feed_dict={model.input_holder: img_flip,
            #                                    model.label_holder: label_flip,
            #                                    model.training: True})
            #     whole_loss += loss
            #     whole_acc += acc
            #     count = count + 1
            # if(i%4==2):
            #     # reverse img
            #     img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
            #     label = cv2.resize(label, (label_size, label_size))
            #     label = label.astype(np.float32) / 255.
            #     img = - img
            #     img = img.reshape((1, img_size, img_size, 3))
            #     label = np.stack((label, 1 - label), axis=2)
            #     label = np.reshape(label, [-1, 2])
            #     _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
            #                         feed_dict={model.input_holder: img,
            #                                    model.label_holder: label,
            #                                    model.training: True})
            #     whole_loss += loss
            #     whole_acc += acc
            #     count = count + 1

            if count % 200 == 0:
                print("v79 Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count)),file=f)
                f.flush()

        saver.save(sess, '/mnt/hdd1/zhengtao/bi_sq/V79_3_Model/model.ckpt', global_step=i)
        
        prec = []
        recall = []  
        for f_img, f_label in zip(ecssd_list,labels_ecssd):

            img = cv2.imread(f_img).astype(np.float32)
            img_shape = img.shape

            label_gt = cv2.imread(f_label)[:, :, 0].astype(np.float32)


            img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
            label = cv2.resize(label_gt, (label_size, label_size))
            label = label.astype(np.float32) / 255.

            img = img.reshape((1, img_size, img_size, 3))
            label = np.stack((label, 1-label), axis=2)
            label = np.reshape(label, [-1, 2])

            loss, acc, result = sess.run([model.Loss_Mean, model.accuracy,model.Prob],
                                    feed_dict={model.input_holder: img,
                                               model.label_holder: label,
                                               model.training: False})
            result = np.reshape(result, (label_size, label_size, 2))
            result = result[:, :, 0]
            result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))
            label_gt=label_gt.astype(np.float32) / 255
            curPrec, curRecall=evaluate.PR_Curve(result,label_gt)
            prec.append(curPrec)
            recall.append(curRecall)


            ecssd_loss += loss
            ecssd_acc += acc
            ecssd_count = ecssd_count + 1

            # add horizon flip image for training

            if ecssd_count==len(ecssd_list):
                print("v79 Epoch ECSSD %d:loss %f, Accuracy %f" % (i, (ecssd_loss/len(ecssd_list)),(ecssd_acc/len(ecssd_list))),file=f)
                f.flush()
        prec = np.hstack(prec[:])
        recall = np.hstack(recall[:])
        prec = np.mean(prec, 1)
        recall = np.mean(recall, 1)
        score = (1+np.sqrt(0.3)**2)*prec*recall / (np.sqrt(0.3)**2*prec + recall)
        curScore = np.max(score)
        print("v79 Epoch %d:F_measure %f" % (i, curScore),file=f)
        
        print("v79 Epoch %d:loss %f, Accuracy %f" % (i, (whole_loss/len(train_list)),(whole_acc/len(train_list))),file=f)
        f.flush()
    f.close()
        
    
    
