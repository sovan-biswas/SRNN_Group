import matplotlib.pyplot as plt
import datetime
import os

import tensorflow as tf

import volleyball
from volleyball import *
from experimentconfig import *
from dataGenerator_exp3 import *
from architecture_exp4 import *
import time

config = ExperimentConfig()

## reading the dataset
train = volley_read_dataset(config.data_path, TRAIN_SEQS + VAL_SEQS)
train_frames = volley_all_frames(train)
test = volley_read_dataset(config.data_path, TEST_SEQS)
test_frames = volley_all_frames(test)

### Data Generator
#train_frames= [(0, 3596),(0, 58376),(0, 3596),(0, 3596),(0, 3596),(0, 3596),(0, 3596),(0, 3596),(0, 3596),(0, 3596)]
trainData = DataGenerator(train_frames, train, True, toShuffle = True)
#testData = DataGenerator(test_frames, test, False)
#start_time = time.time()
#b = trainData.get_batch()
#print len(b), len(b[0])
#print("--- %s seconds ---" % (time.time() - start_time))
#raise Exception("M Done")

model = CompleteModel()
#model.training()
model.main_model()
#raise Exception("M Done")

tf_config = tf.ConfigProto()
#tf_config.log_device_placement = True
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.98

np.random.seed(config.train_random_seed)
tf.set_random_seed(config.train_random_seed)


if config.training_type == 1:
    # Add the loss to summary
    tf.summary.scalar('Loss', model.loss)

    # Add the accuracy to the summary
    tf.summary.scalar('Accuracy', model.accuracy)

    fetch_out = [
        model.loss,
        model.accuracy
    ]

elif config.training_type == 2 or config.training_type == 5:
    # Add the loss to summary
    tf.summary.scalar('Local Loss', model.local_loss)

    # Add the accuracy to the summary
    tf.summary.scalar('Local Accuracy', model.local_accuracy)

    fetch_out = [
        model.loss,
        model.local_accuracy
    ]

elif config.training_type == 3:
    # Add the loss to summary
    tf.summary.scalar('Global Loss', model.global_loss)

    # Add the accuracy to the summary
    tf.summary.scalar('Global Accuracy', model.global_accuracy)

    fetch_out = [
        model.loss,
        model.global_accuracy
    ]

elif config.training_type == 4 or config.training_type == 6:

    # Add the loss to summary
    tf.summary.scalar('Loss', model.loss)

    # Add the loss to summary
    tf.summary.scalar('Local Loss', model.local_loss)

    # Add the loss to summary
    tf.summary.scalar('Global Loss', model.global_loss)

    # Add the accuracy to the summary
    tf.summary.scalar('Local Accuracy', model.local_accuracy)

    # Add the accuracy to the summary
    tf.summary.scalar('Global Accuracy', model.global_accuracy)

    fetch_out = [
        model.loss,
        model.local_loss,
        model.local_accuracy,
        model.global_loss,
        model.global_accuracy
    ]

else:
    raise Exception("Unknown list of training type option")

#with tf.device('/cpu:0'):
#
#    # Add gradients to summary
#    for gradient, var in model.gradients:
#        tf.summary.histogram(var.name, var)
#        tf.summary.histogram(var.name + '/gradient', gradient)

extension_path = '/{0}/{1}/'.format(config.feature_options[config.feature_type - 1],
                                    config.training_options[config.training_type])

dir = os.path.dirname(config.tensorboard_log + extension_path)

if not os.path.exists(dir):
    os.makedirs(dir)
#varrr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# Initialize the FileWriter
writer = tf.summary.FileWriter(config.tensorboard_log + extension_path)

# Merge all summaries together
fetch_out.append(tf.summary.merge_all())


with tf.Session(config=tf_config) as sess:
    # Initialize all variables

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                           max_to_keep=config.train_max_to_keep)

    ## Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    dir = os.path.dirname(config.checkpoint_path + extension_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    path_name = tf.train.latest_checkpoint(config.checkpoint_path + extension_path)
    print ("Path Name = %s" % path_name)

    if config.feature_type == 1 and (config.training_type == 1 or config.training_type == 5
                                     or config.training_type == 6):

        if path_name == None:
            ## Load the pretrained weights into the non-trainable layer
            model.model.load_initial_weights(sess)         # if training from scratch
            start_epoch = 0
        else:
            start_epoch = int(path_name.split('-')[1])
            saver.restore(sess, path_name)
            print path_name, start_epoch

    elif config.feature_type == 1 and (config.training_type == 2 or config.training_type == 3
                                     or config.training_type == 4):

        if path_name == None:
            ## Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(config.alexnet_weights_path,sess)         # if training from scratch
            start_epoch = 0
        else:
            start_epoch = int(path_name.split('-')[1])
            saver.restore(sess, path_name)
            print path_name, start_epoch

    elif config.feature_type == 2:
        start_epoch = int(path_name.split('-')[1])
        saver.restore(sess, path_name)
        print path_name, start_epoch

    elif config.feature_type == 4:
        start_epoch = int(path_name.split('-')[1])
        saver.restore(sess, path_name)
        print path_name, start_epoch

    step = 10
    sess.run(model.zero_ops)
    forward_pass_count = 0
    for epoch in range(start_epoch+step,config.train_num_steps+1,step):
        start_time = time.time()
        batch_lst = trainData.get_batch(step)
        load_time = (time.time() - start_time)
        start_time2 = time.time()
        #pool = ThreadPool(config.num_cpus)
        #out = pool.map(forward_compute, batch_lst)
        #pool.close()
        #pool.join()
        for batch in batch_lst:

            if config.feature_type == 2 and config.training_type == 1:
                feed_dict = {model.x: batch[0],
                             model.y: batch[1],
                             model.keep_prob: 0.5}
            elif config.feature_type == 2 and (config.training_type == 2 or config.training_type == 5):
                feed_dict = {model.x_series: batch[0],
                             model.locs: batch[1],
                             model.l_label: batch[2],
                             model.y: batch[3],
                             model.keep_prob: 0.5}

            elif config.feature_type == 2 and config.training_type == 3:
                feed_dict = {model.x_series: batch[0],
                             model.locs: batch[1],
                             model.l_label: batch[2],
                             model.y_global: batch[3],
                             model.label: batch[4],
                             model.keep_prob: 0.5}

            elif config.feature_type == 2 and (config.training_type == 4 or config.training_type == 6):
                feed_dict = {model.x_series: batch[0],
                             model.locs: batch[1],
                             model.l_label: batch[2],
                             model.y: batch[3],
                             model.y_global: batch[4],
                             model.label: batch[5],
                             model.keep_prob: 0.5}
            elif config.feature_type == 4 and (config.training_type == 2 or config.training_type == 5):
                feed_dict = {model.x_series: batch[0],
                             model.y: batch[1],
                             model.keep_prob: 0.5,
                             model.is_training: True}
            elif config.feature_type == 4 and config.training_type == 3:
                feed_dict = {model.x_series: batch[0],
                             model.y_global: batch[1],
                             model.label: batch[2],
                             model.keep_prob: 0.5,
                             model.is_training: True}
            elif config.feature_type == 4 and (config.training_type == 4 or config.training_type == 6):
                feed_dict = {model.x_series: batch[0],
                             model.y: batch[1],
                             model.y_global: batch[2],
                             model.label: batch[3],
                             model.keep_prob: 0.5,
                             model.is_training: True}
            elif config.feature_type == 1 and (config.training_type == 2 or config.training_type == 5):
                feed_dict = {model.x_series: batch[0],
                             model.y: batch[1],
                             model.keep_prob: 0.5}
            elif config.feature_type == 1 and config.training_type == 3:
                feed_dict = {model.x_series: batch[0],
                             model.y_global: batch[1],
                             model.label: batch[2],
                             model.keep_prob: 0.5}
            elif config.feature_type == 1 and (config.training_type == 4 or config.training_type == 6):
                feed_dict = {model.x_series: batch[0],
                             model.y: batch[1],
                             model.y_global: batch[2],
                             model.label: batch[3],
                             model.keep_prob: 0.5}

            if config.isSRNN:
                feed_dict[model.edge_x] = batch[-2]
                feed_dict[model.edge_label] = batch[-1]
                #feed_dict[model.edge_x] = batch[-1]

            #[v1,v2,v3,v4,v5,v6] = sess.run([model.deb1,model.deb2,model.deb3,model.deb4,model.deb5,model.deb6],feed_dict)
            #print v1, v2,v3,v4,v5,v6
            #raise  Exception("Done")
            sess.run(model.accum_ops,feed_dict)
            forward_pass_count += 1
            if forward_pass_count == config.num_steps_for_back_propagation:
                #print ("Backward Pass")
                sess.run(model.train_op)
                sess.run(model.zero_ops)
                forward_pass_count = 0
        #raise Exception("Done")
        batch = batch_lst[-1]
        del batch_lst
        process_time = (time.time() - start_time2)

        if (epoch-start_epoch) % 50 == 0:
            #start_time2 = time.time()
            outputs = sess.run(fetch_out,
                               feed_dict)
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if config.training_type in [1, 2, 3, 5]:
                print('--- %s Step:%d ' % (ts, epoch) +
                      'total.L: %.4f ' % (outputs[0]) +
                      'total.A: %.4f' % (outputs[1]))
            elif config.training_type in [4, 6]:
                print('--- %s Step:%d ' % (ts, epoch) +
                      'total.L: %.4f ' % (outputs[0]) +
                      'Node.L: %.4f ' % (outputs[1]) +
                      'Node.A: %.4f ' % (outputs[2]) +
                      'Global.L: %.4f ' % (outputs[3]) +
                      'Global.A: %.4f' % (outputs[4]))

            feed_dict[model.keep_prob] = 1.
            if config.feature_type == 4:
                feed_dict[model.model.is_training] = False

            outputs = sess.run(fetch_out,
                               feed_dict)

            writer.add_summary(outputs[-1], epoch)
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if config.training_type in [1, 2, 3, 5]:
                print('--- %s Step:%d ' % (ts, epoch) +
                      'total.L: %.4f ' % (outputs[0]) +
                      'total.A: %.4f' % (outputs[1]))
            elif config.training_type in [4, 6]:
                print('--- %s Step:%d ' % (ts, epoch) +
                      'total.L: %.4f ' % (outputs[0]) +
                      'Node.L: %.4f ' % (outputs[1]) +
                      'Node.A: %.4f ' % (outputs[2]) +
                      'Global.L: %.4f ' % (outputs[3]) +
                      'Global.A: %.4f' % (outputs[4]))


            #print("--- Evaluation %s seconds ---" % (time.time() - start_time2))
        print("One Step %.2f secs: (L: %.2f secs, P: %.2f secs)" % (time.time() - start_time, load_time, process_time))
        if (epoch-start_epoch) % config.train_save_every_steps == 0:
            print('saving the model at %d steps' % epoch)
            saver.save(sess, config.checkpoint_path + extension_path + 'model', epoch, write_meta_graph=True)
print('done!')
