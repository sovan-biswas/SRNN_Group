import matplotlib.pyplot as plt
import datetime
import os

import tensorflow as tf

import volleyball
from volleyball import *
from experimentconfig import *
#from dataGenerator_inception import *
#from architecture_exp4 import *
from dataGenerator_exp3 import *
from architecture_exp4 import *
import time

config = ExperimentConfig()

## reading the dataset
test = volley_read_dataset(config.data_path, TEST_SEQS)
test_frames = volley_all_frames(test)

### Data Generator

#trainData = DataGenerator(train_frames, train, True, toShuffle = True)

#start_time = time.time()
#b = trainData.get_batch()
# print len(b)
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

extension_path = '/{0}/{1}/'.format(config.feature_options[config.feature_type - 1],
                                    config.training_options[config.training_type])

#for val in [5800,6300,6800,7300,7800,8300,8800,9300,9800,10300]:
for val in [3600, 2700, 1800, 900]:
#for val in range(6050,10801,250):
#for val in range(8000,799,-800):
    testData = DataGenerator(test_frames, test, False, toShuffle=False)
    with tf.Session(config=tf_config) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                               max_to_keep=config.train_max_to_keep)
        path_name = config.checkpoint_path + extension_path + 'model-{0}'.format(val)
        saver.restore(sess,path_name)

        loss = 0
        local_accuracy = 0
        global_accuracy = 0
        global_loss = 0
        count = 0
        step = 200
        for epoch in range(0,1401,step):
            start_time = time.time()
            batch_lst = testData.get_batch(step)
            #print("--- Data Load %s seconds ---" % (time.time() - start_time))
            for batch in batch_lst:
                start_time2 = time.time()
                if config.feature_type == 1 and config.training_type == 1:
                    feed_dict = {model.x: batch[0],
                                 model.y: batch[1],
                                 model.keep_prob: 1.0}
                    outputs = sess.run([model.loss, model.accuracy],feed_dict)
                elif config.feature_type == 2:
                    feed_dict = {model.x_series: batch[0],
                             model.locs: batch[1],
                             model.l_label: batch[2],
                             model.y: batch[3],
                             model.y_global: batch[4],
                             model.label: batch[5],
                             model.keep_prob: 0.5}
                    if config.isSRNN:
                        feed_dict[model.edge_x] = batch[-2]
                        feed_dict[model.edge_label] = batch[-1]
                elif config.feature_type == 4:
                    feed_dict = {model.x_series: batch[0],
                                model.y: batch[1],
                                model.y_global: batch[2],
                                model.label: batch[3],
                                model.keep_prob: 0.5,
                                model.is_training: False}
                    if config.isSRNN:
                        feed_dict[model.edge_x] = batch[-2]
                        feed_dict[model.edge_label] = batch[-1]
                    outputs = sess.run([model.local_loss,model.local_accuracy,model.global_loss,model.global_accuracy],
                                       feed_dict)
                loss += outputs[0]
                local_accuracy += outputs[1]
                global_loss += outputs[2]
                global_accuracy += outputs[3]
                count += 1

            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s Step ' % (ts) +
                  'node.L: %.4f ' % (loss/count) +
                  'node.A: %.4f ' % (local_accuracy/count) +
                  'global.L: %.4f ' % (global_loss / count) +
                  'global.A: %.4f ' %(global_accuracy / count))

        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('%s Average Total ' % (ts) +
              'node.L: %.4f ' % (loss / count) +
              'node.A: %.4f ' % (local_accuracy / count) +
              'global.L: %.4f ' % (global_loss / count) +
              'global.A: %.4f ' %(global_accuracy / count))
        print('done!')
