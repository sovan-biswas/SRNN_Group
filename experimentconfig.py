import os

# Experimentation parameters
class ExperimentConfig(object):

    def __init__(self):

        #experiment setup
        self.feature_options = ['Alexnet','InceptionV3', 'I3D', 'VGG']
        self.feature_type = 1 #options are any one of the above
        self.training_options =['test','fineTuneFeatures','ActionTraining', 'ActivityTraining',
                                'ActionTraining+ActivityTraining', 'fineTuneFeature+ActionTraining',
                                'fineTuneFeature+ActionTraining+ActivityTraining']
        self.training_type = 6 ### use 0 to just experiment or test a sample

        self.isSRNN = True
        self.group_split = 2


        self.data_augementation = True
        self.num_cpus = 10 ### Number of cup cores
        self.parallel_loader = True
        self.num_steps_for_back_propagation = 3

        # shared
        self.image_size = 720, 1280
        self.temporal_size = 10
        self.out_size = 87, 157
        self.batch_size = 1  ## change this
        self.num_boxes = 12
        self.epsilon = 1e-9

        # Alexnet parameters
        self.alexnet_size = 227, 227
        self.alexnet_keep_prob = 0.5 #### check there is train dropout probability
        self.alexnet_train_layers = ['fc8', 'fc7', 'fc6']
        self.alexnet_num_classes = 9
        self.alexnet_weights_path = './models/preTrainedModels/bvlc_alexnet.npy'

        # LSTM parameters
        self.lstm_size = 2000
        self.global_lstm_size = 1000
        self.edge_lstm_size = 24
        self.node_sliding_window = 5
        self.global_sliding_window = 10
        self.max_nodes = 12
        self.features_multiscale_names = ['Mixed_5d', 'Mixed_6e']
        self.train_inception = False

        # DetNet
        self.build_detnet = False
        self.num_resnet_blocks = 1
        self.num_resnet_features = 512
        self.reg_loss_weight = 10.0
        self.nms_kind = 'greedy'

        # ActNet
        self.use_attention = False
        self.crop_size = 5, 5
        self.num_features_boxes = 4096
        self.num_actions = 9
        self.num_activities = 8
        self.actions_loss_weight = 1.0#4.0
        self.actions_weights = [[1., 1., 2., 6., 0.5, 2., 2., 0.04, 1]]
        #self.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]
        #self.actions_weights = [[1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        # sequence
        self.num_features_hidden = 1024
        self.num_frames = 10
        self.num_before = 5
        self.num_after = 4

        # training parameters
        self.train_num_steps = 7200
        self.train_random_seed = 0
        self.train_learning_rate = 0.00001
        self.train_learning_rate = 0.00001
        self.train_dropout_prob = 0.8
        self.train_save_every_steps = 900
        self.train_max_to_keep = 20


        #Paths
        # NOTE: you have to fill this
        self.model_folder = '/media/data/sovan/codes/sovan_code/volleyball_0711/models'
        self.data_path = '/media/data/sovan/data/Volleyball/volleyball/'
        self.checkpoint_path = '/media/data/sovan/codes/sovan_code/volleyball_0711/checkpoints_test'
        self.tensorboard_log = '/media/data/sovan/codes/sovan_code/volleyball_0711/tensorboard_test'
        # reading images of a certain resolution
        self.images_path = self.data_path

        ## Load models Path
        if self.feature_type == 1: ## Alexnet
            self.model_path = self.model_folder + '/preTrainedModels/bvlc_alexnet.npy'
        elif self.feature_type == 2: ## ImageNetV3
            # you can download pre-trained models at
            # http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
            self.model_path = self.model_folder + '/preTrainedModels/inception_v3.ckpt'
        elif self.feature_type == 3: ## I3D
            self.model_path = self.model_folder + '/preTrainedModels/i3d'
            self.flow_path = self.model_folder + '/preTrainedModels/i3d'
        elif self.feature_type == 4: ## I3D
            self.model_path = self.model_folder + '/preTrainedModels/i3d'
        else:
            raise Exception("Not supported Model Type")
