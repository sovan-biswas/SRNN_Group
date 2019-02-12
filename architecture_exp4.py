from experimentconfig import *
from models.alexnet import *
#from models.i3d import *
from models.inceptionV3 import *
from models.vgg16 import *
import tensorflow as tf
from math import ceil

class CompleteModel:
    def __init__(self):

        self.config = ExperimentConfig()
        if self.config.feature_type == 1:
            if self.config.training_type == 1:
                self.x = tf.placeholder(tf.float32, [None, self.config.alexnet_size[0],
                                                     self.config.alexnet_size[1], 3])  # (batch_size,227,227,3)
            # elif self.config.training_type == 2 or self.config.training_type == 5 or self.config.training_type == 3:
            else:
                self.x_series = tf.placeholder(tf.int32, [None, self.config.temporal_size,
                                                          self.config.alexnet_size[0],
                                                          self.config.alexnet_size[1], 3])
                self.x = tf.reshape(tf.cast(self.x_series, tf.float32), [-1, self.config.alexnet_size[0],
                                                                         self.config.alexnet_size[1], 3])
            self.keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('Alexnet') as scope:
                self.model = AlexNet(self.x, self.keep_prob,
                                     self.config.alexnet_num_classes, ['fc8'],
                                     weights_path=self.config.alexnet_weights_path)

        elif self.config.feature_type == 2:
            self.x_series = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.temporal_size, 720, 1280, 3))
            self.x = tf.reshape(tf.cast(self.x_series, tf.float32), [-1, 720, 1280, 3])
            self.locs = tf.placeholder(tf.float32, shape=(None, self.config.temporal_size, 4))
            self.l_label = tf.placeholder(tf.int32, [None, ])
            self.keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('InceptionV3') as scope:
                self.model = InceptionV3(self.x, self.keep_prob)

            locs = tf.reshape(self.locs,[-1,4])
            inds = tf.reshape(self.l_label,[-1,])

            x = tf.image.crop_and_resize(self.model.feat_map, locs, inds, tf.constant([3,2]))

            #x = tf.contrib.slim.avg_pool2d(x, [3, 2], scope='average_pool')
            x = tf.contrib.layers.flatten(x)

            x = self.fc_layer(x, 6*2048, 2048, 'fc')

            self.x = tf.reshape(x,[-1,2048])

        elif self.config.feature_type == 3:
            print ("Config 3")
            # self.x_series = tf.placeholder(tf.float32, shape=(None, self.config.temporal_size, 1280, 720, 3))
            # self.x_flow = tf.placeholder(tf.float32, shape=(None, self.config.temporal_size, 1280, 720, 3))
            # self.locs = tf.placeholder(tf.float32, shape=(None, self.config.temporal_size, 4))
            # self.l_label = tf.placeholder(tf.bool, [None, self.config.batch_size])
            # self.keep_prob = tf.placeholder(tf.float32)
            # with tf.name_scope('I3D') as scope:
            #     with tf.variable_scope('RGB'):
            #         rgb_model = InceptionI3d(400, spatial_squeeze=True, final_endpoint='Mixed_5c')
            #         rgb_logits, _ = rgb_model(self.x_series, is_training=False, dropout_keep_prob=1.0)
            #     with tf.variable_scope('Flow'):
            #         flow_model = InceptionI3d(400, spatial_squeeze=True, final_endpoint='Mixed_5c')
            #         flow_logits, _ = flow_model(self.x_flow, is_training=False, dropout_keep_prob=1.0)
            # rgb_variable_map = {}
            # for variable in tf.global_variables():
            #     if variable.name.split('/')[0] == 'RGB':
            #         rgb_variable_map[variable.name.replace(':0', '')] = variable
            #
            # flow_variable_map = {}
            # for variable in tf.global_variables():
            #     if variable.name.split('/')[0] == 'Flow':
            #         flow_variable_map[variable.name.replace(':0', '')] = variable
            #
            # ### Process the rgb_logits and flow_logits to retrive sequence of features
            # self.model = self.process_logic(rgb_logits,flow_logits, self.locs, self.l_label)
        elif self.config.feature_type == 4:
            if self.config.training_type == 1:
                self.x = tf.placeholder(tf.float32, [None, self.config.alexnet_size[0],
                                                     self.config.alexnet_size[1], 3])  # (batch_size,227,227,3)
            # elif self.config.training_type == 2 or self.config.training_type == 5 or self.config.training_type == 3:
            else:
                self.x_series = tf.placeholder(tf.int32, [None, self.config.temporal_size,
                                                          self.config.alexnet_size[0],
                                                          self.config.alexnet_size[1], 3])
                self.x = tf.reshape(tf.cast(self.x_series, tf.float32), [-1, self.config.alexnet_size[0],
                                                                         self.config.alexnet_size[1], 3])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            self.model = VGG(self.x, self.keep_prob, self.is_training)

        else:
            raise Exception("Not supported Feature Type")



    def load_initial_weights(self, path, session):
        #### Loading initial Imagenet pre-trained weights
        variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        # Load the weights into memory
        weights_dict = np.load(path).item()

        # Loop over all layer names stored in the weights dict
        for op_name in variables.keys():

            # Check if the layer is one of the layers that should be reinitialized
            if op_name in weights_dict.keys():
                data = weights_dict[op_name]
                session.run(variables[op_name].assign(data))

    def sliding_window_block(self, data, sliding_window, name):
        with tf.name_scope(name) as scope:
            # sliding window
            pad_val = tf.tile(data[:, 0:1, :], [1, sliding_window - 1, 1], name=scope + '_tile')
            #pad_feat = tf.concat(1, [pad_val, data], name=scope + '_concat1')
            pad_feat = tf.concat([pad_val, data], 1, name=scope + '_concat1')

            indx = range(0, self.config.temporal_size)
            # indx = tf.range(0, self.config.temporal_size, 1)
            # process_data = tf.concat(0, [pad_feat[:, i:i + sliding_window, :] for i in indx],
            #                         name=scope + '_concat2')
            process_data = tf.concat([pad_feat[:, i:i + sliding_window, :] for i in indx],0,
                name = scope + '_concat2')
        return process_data

    def lstm_block(self, data, lstm_size, name):
        with tf.name_scope(name) as scope:
            ### start defining LSTM model
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
            #outputs, _ = tf.nn.rnn(lstm, data, dtype=tf.float32, scope=name)
            outputs, _ = tf.nn.static_rnn(lstm, data, dtype=tf.float32, scope=name)

            lstm_output = outputs[-1]
        return lstm_output

    def lstm_layer(self, data, lstm_size, sliding_window, name):

        data_x = self.sliding_window_block(data, sliding_window, name + '_sliding_window')
        data_lst = tf.unstack(data_x, sliding_window, 1, name=name + '_unstack')

        lstm_output = self.lstm_block(data_lst, lstm_size, name + '_lstm_block')
        return lstm_output

    def fc_layer(self, data, fc_in, fc_out, name):

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[fc_in, fc_out], trainable=True)
            biases = tf.get_variable('biases', [fc_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            score = tf.nn.xw_plus_b(data, weights, biases, name=scope.name)

        return score

    def max_pool_layer(self, data, lstm_size, mask, name):

        with tf.variable_scope(name) as scope:
            data = tf.transpose(
                tf.reshape(data, [self.config.temporal_size, -1, lstm_size],
                           name=scope.name + '_reshape'), perm=[1, 0, 2], name=scope.name + '_transpose')
            umask = tf.unstack(mask, self.config.batch_size * self.config.group_split, 1, name=scope.name + '_unstack1')
            data_lst = [tf.reduce_max(tf.boolean_mask(data, l, name=name + '_mask'), 0, name=scope.name + '_maxpool')
                        for l in umask]

            data_x = tf.stack(data_lst, 0, name=scope.name + '_stack')

            data_x = tf.reshape(data_x,
                                [self.config.batch_size, self.config.group_split, self.config.temporal_size, -1],
                                name=scope.name + '_reshape2')
            #data_x = tf.concat(2, tf.unstack(data_x, self.config.group_split, 1, name=scope.name + '_unstack2'))
            data_x = tf.concat(tf.unstack(data_x, self.config.group_split, 1, name=scope.name + '_unstack2'), 2)

        return data_x

    def edge_pooling_layer(self, data, lstm_size, edgemask, name):
        with tf.name_scope(name) as scope:
            data = tf.transpose(
                tf.reshape(data, [self.config.temporal_size, -1, lstm_size],
                           name=scope + '_reshape'), perm=[1, 0, 2], name=scope + '_transpose')
            edge_mask = tf.unstack(edgemask, 8, 3)
            edge_pool = []
            for iter_orient in range(8):
                edge_mask_orient = tf.unstack(edge_mask[iter_orient], self.config.temporal_size, 2)
                edge_pool_quad = []
                for iter_temp in range(self.config.temporal_size):
                    x_edge_process = tf.stack([tf.reduce_sum(tf.boolean_mask(data[:, iter_temp, :], m), axis=0) for m in
                                               tf.unstack(edge_mask_orient[iter_temp],
                                                          self.config.batch_size * self.config.max_nodes, 1)], 0)
                    edge_pool_quad.append(x_edge_process)
                edge_pool.append(tf.stack(edge_pool_quad,1))
            #edge_pool = tf.concat(2,edge_pool)
            edge_pool = tf.concat(edge_pool,2)
            final_pool = tf.stack(tf.boolean_mask(edge_pool,tf.reduce_any(edgemask, [0, 2, 3])),0)

            return final_pool

    def main_model(self):

        if self.config.feature_type == 1 or self.config.feature_type == 4:

            # input = self.preprocess_alexnet(input_images, input_box)
            if self.config.training_type == 1:  ### Finetunes the alexnet only

                self.y = tf.placeholder(tf.bool, [None, self.config.num_actions])
                y = tf.cast(self.y, tf.float32)
                score = self.model.fc8
                # list of training variables
                var_list = [v for v in tf.trainable_variables() if
                            v.name.split('/')[0] in self.config.alexnet_train_layers]

                # Op for calculating the loss
                with tf.name_scope("cross_ent"):
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.y))
                    predictions = tf.nn.softmax(score, name='preds')
                    self.loss = -tf.reduce_mean(tf.constant(self.config.actions_weights) *
                                                y * tf.log(predictions + self.config.epsilon))

                # Train op
                with tf.name_scope("train"):
                    # Get gradients of all trainable variables
                    gradients = tf.gradients(self.loss, var_list)
                    self.gradients = list(zip(gradients, var_list))

                    # Create optimizer and apply gradient descent to the trainable variables
                    optimizer = tf.train.GradientDescentOptimizer(self.config.train_learning_rate)
                    self.train_op = optimizer.apply_gradients(grads_and_vars=self.gradients)
                # Evaluation op: Accuracy of the model
                with tf.name_scope("accuracy"):
                    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(self.y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                    # input_images = tf.boolean_mask(input_images, image_sample, name='images')
                    # input_action = tf.boolean_mask(input_action, image_sample, name='image_label')

            else:
                ### SRNN and Hierarchical RNN
                self.y = tf.placeholder(tf.bool, [None, self.config.num_actions])
                self.label = tf.placeholder(tf.bool, [None, self.config.batch_size * self.config.group_split])
                self.y_global = tf.placeholder(tf.bool, [self.config.batch_size, self.config.num_activities])
                if self.config.isSRNN:
                    self.edge_x = tf.placeholder(tf.float16, [None, self.config.temporal_size, 36])
                    self.edge_label = tf.placeholder(tf.bool, [None, self.config.batch_size * self.config.max_nodes,
                                                               self.config.temporal_size, 8])
                with tf.name_scope('Nodes') as scope:
                    feat = tf.reshape(self.model.feat, [-1, self.config.temporal_size, 4096])
                    if self.config.isSRNN:
                        ### SRNN with edge
                        edge_x = tf.cast(self.edge_x, dtype=tf.float32)
                        x_edge = self.lstm_layer(edge_x, self.config.edge_lstm_size, self.config.node_sliding_window,
                                                 'edge')

                        ### Dropout Layer (Edge)
                        #x_edge = tf.nn.dropout(x_edge, self.keep_prob)

                        x_edge = self.edge_pooling_layer(x_edge, self.config.edge_lstm_size, self.edge_label, 'edgepool')

                        #feat = tf.concat(2, [feat, x_edge])
                        feat = tf.concat([feat, x_edge],2)
                    indx = range(0, self.config.temporal_size)
                    y = tf.concat([tf.cast(self.y, tf.float32) for i in indx],0)

                    ### LSTM Layer (Node)
                    x_node = self.lstm_layer(feat, self.config.lstm_size,
                                             self.config.node_sliding_window, 'node')

                    ### Dropout Layer (Node)
                    node_output = tf.nn.dropout(x_node, self.keep_prob)

                    ### FC Layer (Node)
                    local_score = self.fc_layer(node_output, self.config.lstm_size, self.config.num_actions,
                                                'node_fc_layer')

                ### Max Pooling (Video)
                v_x = self.max_pool_layer(x_node, self.config.lstm_size, self.label, name='pool')

                with tf.name_scope('Global') as scope:
                    ### LSTM Layer (Global)
                    x_global = self.lstm_layer(v_x, self.config.global_lstm_size,
                                               self.config.global_sliding_window, 'global')

                    ### global label
                    y_global = tf.concat([tf.cast(self.y_global, tf.float32) for i in indx],0)

                    ### Dropout Layer (Global)
                    video_output = tf.nn.dropout(x_global, self.keep_prob)

                    ### FC Layer (Global)
                    global_score = self.fc_layer(video_output, self.config.global_lstm_size,
                                                 self.config.num_activities, 'global_fc_layer')

                ### Losses:
                # Op for calculating the loss
                with tf.name_scope("local_cross_ent"):
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.y))

                    predictions = tf.nn.softmax(local_score, name='preds')
                    self.local_loss = -tf.reduce_mean(tf.constant(self.config.actions_weights) *
                                                      y * tf.log(predictions + self.config.epsilon))

                # Evaluation op: Accuracy of the model
                with tf.name_scope("local_accuracy"):
                    correct_pred = tf.equal(tf.argmax(local_score, 1), tf.argmax(y, 1))
                    self.local_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                # Op for calculating the loss
                with tf.name_scope("global_cross_ent"):
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.y))

                    global_predictions = tf.nn.softmax(global_score, name='global_preds')
                    self.global_loss = -tf.reduce_mean(y_global *
                                                       tf.log(global_predictions + self.config.epsilon))

                # Evaluation op: Accuracy of the model
                with tf.name_scope("global_accuracy"):
                    global_correct_pred = tf.equal(tf.argmax(global_score, 1), tf.argmax(y_global, 1))
                    self.global_accuracy = tf.reduce_mean(tf.cast(global_correct_pred, tf.float32))

                #### trainable variable
                if self.config.training_type == 2:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['edge_lstm_block', 'node_lstm_block', 'node_fc_layer']]
                    self.loss = self.local_loss

                elif self.config.training_type == 3:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['global_lstm_block', 'global_fc_layer']]
                    self.loss = self.global_loss

                elif self.config.training_type == 4:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['edge_lstm_block', 'node_lstm_block', 'node_fc_layer',
                                                         'global_lstm_block', 'global_fc_layer']]
                    self.loss = self.config.actions_loss_weight * self.local_loss + self.global_loss

                elif self.config.training_type == 5:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['fc7', 'fc6', 'edge_lstm_block', 'node_lstm_block',
                                                         'node_fc_layer']]
                    self.loss = self.local_loss

                elif self.config.training_type == 6:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['fc7', 'fc6', 'edge_lstm_block', 'node_lstm_block',
                                                         'node_fc_layer', 'global_lstm_block', 'global_fc_layer']]
                    self.loss = self.config.actions_loss_weight * self.local_loss + self.global_loss

                else:
                    raise Exception("Unknown list of training type option")

                # Train op
                with tf.name_scope("train"):
                    # Get gradients of all trainable variables
                    # self.var_list = []
                    # Create optimizer and apply gradient descent to the trainable variables
                    #optimizer = tf.train.AdamOptimizer(
                    #    self.config.train_learning_rate)  # tf.train.GradientDescentOptimizer(self.config.train_learning_rate)
                    optimizer = tf.train.GradientDescentOptimizer(self.config.train_learning_rate)
                    self.gradients = optimizer.compute_gradients(self.loss, var_list)

                    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                                  var_list]
                    self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

                    self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gradients)]
                    self.train_op = optimizer.apply_gradients(
                        [(accum_vars[i], gv[1]) for i, gv in enumerate(self.gradients)])

        elif self.config.feature_type == 2 :

            # input = self.preprocess_alexnet(input_images, input_box)
            if self.config.training_type == 1:  ### Finetunes the alexnet only

                self.y = tf.placeholder(tf.bool, [None, self.config.num_actions])
                y = tf.cast(self.y, tf.float32)
                score = self.model.fc8
                # list of training variables
                var_list = [v for v in tf.trainable_variables() if
                            v.name.split('/')[0] in self.config.alexnet_train_layers]

                # Op for calculating the loss
                with tf.name_scope("cross_ent"):
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.y))
                    predictions = tf.nn.softmax(score, name='preds')
                    self.loss = -tf.reduce_mean(tf.constant(self.config.actions_weights) *
                                                y * tf.log(predictions + self.config.epsilon))

                # Train op
                with tf.name_scope("train"):
                    # Get gradients of all trainable variables
                    gradients = tf.gradients(self.loss, var_list)
                    self.gradients = list(zip(gradients, var_list))

                    # Create optimizer and apply gradient descent to the trainable variables
                    optimizer = tf.train.AdamOptimizer(
                        self.config.train_learning_rate)  #tf.train.GradientDescentOptimizer(self.config.train_learning_rate)
                    self.train_op = optimizer.apply_gradients(grads_and_vars=self.gradients)
                # Evaluation op: Accuracy of the model
                with tf.name_scope("accuracy"):
                    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(self.y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                    # input_images = tf.boolean_mask(input_images, image_sample, name='images')
                    # input_action = tf.boolean_mask(input_action, image_sample, name='image_label')

            else:
                ### SRNN and Hierarchical RNN
                self.y = tf.placeholder(tf.bool, [None, self.config.num_actions])
                self.label = tf.placeholder(tf.bool, [None, self.config.batch_size * self.config.group_split])
                self.y_global = tf.placeholder(tf.bool, [self.config.batch_size, self.config.num_activities])
                if self.config.isSRNN:
                    self.edge_x = tf.placeholder(tf.float16, [None, self.config.temporal_size, 36])
                    self.edge_label = tf.placeholder(tf.bool, [None, self.config.batch_size * self.config.max_nodes,
                                                               self.config.temporal_size, 8])
                with tf.name_scope('Nodes') as scope:
                    feat = tf.reshape(self.x, [-1, self.config.temporal_size, 2048])
                    if self.config.isSRNN:
                        ### SRNN with edge
                        edge_x = tf.cast(self.edge_x, dtype=tf.float32)
                        x_edge = self.lstm_layer(edge_x, self.config.edge_lstm_size, self.config.node_sliding_window,
                                                 'edge')

                        ### Dropout Layer (Edge)
                        #x_edge = tf.nn.dropout(x_edge, self.keep_prob)

                        x_edge = self.edge_pooling_layer(x_edge, self.config.edge_lstm_size, self.edge_label, 'edgepool')

                        #feat = tf.concat(2, [feat, x_edge])
                        feat = tf.concat([feat, x_edge], 2)
                    indx = range(0, self.config.temporal_size)
                    # y = tf.concat(0, [tf.cast(self.y, tf.float32) for i in indx])
                    y = tf.concat([tf.cast(self.y, tf.float32) for i in indx], 0)

                    ### LSTM Layer (Node)
                    x_node = self.lstm_layer(feat, self.config.lstm_size,
                                             self.config.node_sliding_window, 'node')

                    ### Dropout Layer (Node)
                    node_output = tf.nn.dropout(x_node, self.keep_prob)

                    ### FC Layer (Node)
                    local_score = self.fc_layer(node_output, self.config.lstm_size, self.config.num_actions,
                                                'node_fc_layer')

                ### Max Pooling (Video)
                v_x = self.max_pool_layer(x_node, self.config.lstm_size, self.label, name='pool')

                with tf.name_scope('Global') as scope:
                    ### LSTM Layer (Global)
                    x_global = self.lstm_layer(v_x, self.config.global_lstm_size,
                                               self.config.global_sliding_window, 'global')

                    ### global label
                    y_global = tf.concat([tf.cast(self.y_global, tf.float32) for i in indx], 0)

                    ### Dropout Layer (Global)
                    video_output = tf.nn.dropout(x_global, self.keep_prob)

                    ### FC Layer (Global)
                    global_score = self.fc_layer(video_output, self.config.global_lstm_size,
                                                 self.config.num_activities, 'global_fc_layer')

                ### Losses:
                # Op for calculating the loss
                with tf.name_scope("local_cross_ent"):
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.y))

                    predictions = tf.nn.softmax(local_score, name='preds')
                    self.local_loss = -tf.reduce_mean(tf.constant(self.config.actions_weights) *
                                                      y * tf.log(predictions + self.config.epsilon))

                # Evaluation op: Accuracy of the model
                with tf.name_scope("local_accuracy"):
                    correct_pred = tf.equal(tf.argmax(local_score, 1), tf.argmax(y, 1))
                    self.local_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                # Op for calculating the loss
                with tf.name_scope("global_cross_ent"):
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.y))

                    global_predictions = tf.nn.softmax(global_score, name='global_preds')
                    self.global_loss = -tf.reduce_mean(y_global *
                                                       tf.log(global_predictions + self.config.epsilon))

                # Evaluation op: Accuracy of the model
                with tf.name_scope("global_accuracy"):
                    global_correct_pred = tf.equal(tf.argmax(global_score, 1), tf.argmax(y_global, 1))
                    self.global_accuracy = tf.reduce_mean(tf.cast(global_correct_pred, tf.float32))

                #### trainable variable
                if self.config.training_type == 2:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['edge_lstm_block', 'node_lstm_block', 'node_fc_layer']]
                    self.loss = self.local_loss

                elif self.config.training_type == 3:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['global_lstm_block', 'global_fc_layer']]
                    self.loss = self.global_loss

                elif self.config.training_type == 4:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['edge_lstm_block', 'node_lstm_block', 'node_fc_layer',
                                                         'global_lstm_block', 'global_fc_layer']]
                    self.loss = self.config.actions_loss_weight * self.local_loss + self.global_loss

                elif self.config.training_type == 5:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['InceptionV3', 'fc', 'edge_lstm_block', 'node_lstm_block',
                                                         'node_fc_layer'] or v.name.split('/')[1] in ['Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                                                        'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c' ] ]
                    self.loss = self.local_loss

                elif self.config.training_type == 6:
                    # list of training variables
                    var_list = [v for v in tf.trainable_variables() if
                                v.name.split('/')[0] in ['InceptionV3','fc', 'edge_lstm_block', 'node_lstm_block',
                                                         'node_fc_layer', 'global_lstm_block', 'global_fc_layer']
                                                         or v.name.split('/')[1] in ['Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                                                        'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c' ] ]
                    self.loss = self.config.actions_loss_weight * self.local_loss + self.global_loss

                else:
                    raise Exception("Unknown list of training type option")

                # Train op
                with tf.name_scope("train"):
                    # Get gradients of all trainable variables
                    # self.var_list = []
                    # Create optimizer and apply gradient descent to the trainable variables
                    optimizer = tf.train.AdamOptimizer(
                        self.config.train_learning_rate)  # tf.train.GradientDescentOptimizer(self.config.train_learning_rate)
                    self.gradients = optimizer.compute_gradients(self.loss, var_list)

                    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                                  var_list]
                    self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

                    self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gradients)]

                    self.train_op = optimizer.apply_gradients(
                        [(accum_vars[i], gv[1]) for i, gv in enumerate(self.gradients)])
