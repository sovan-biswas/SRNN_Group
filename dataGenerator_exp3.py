import numpy as np
import pickle

import skimage.io
import skimage.transform
import scipy.misc
from random import shuffle

from experimentconfig import *
from joblib import Parallel, delayed
#from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool
import time


class DataGenerator:

    def __init__(self, frames, anns, isTraining, toShuffle = True):

        self.config = ExperimentConfig()

        # these you can get in the repository
        self.src_image_size = pickle.load(open('./data/src_image_size.p', 'rb'))
        self. all_tracks = pickle.load(open('./data/tracks_normalized.p', 'rb'))

        self.anns = anns
        self.frames = frames
        self.toShuffle = toShuffle
        self.toProcessFrames = self.frames[:]
        if self.toShuffle:
            shuffle(self.toProcessFrames)

        self.isTraining = isTraining

        self.isSRNN = self.config.isSRNN

        if self.config.feature_type == 1:
            print ("Config 1")
        elif self.config.feature_type == 2:
            print ("Config 2")
        elif self.config.feature_type == 3:
            print ("Config 3")
        elif self.config.feature_type == 4:
            print ("Config 4")
        else:
            raise Exception("Not supported Feature Type")

    def get_batch(self,args=1):

        batch_lst = []
        for r in range(args):
            batch_d = [self.toProcessFrames.pop() for i in range(min(self.config.batch_size, len(self.toProcessFrames)))]

            if len(batch_d) < self.config.batch_size and self.isTraining is True:
                self.toProcessFrames = self.frames[:]
                if self.toShuffle:
                    shuffle(self.toProcessFrames)

                batch_d = batch_d + [self.toProcessFrames.pop() for i in range(min(self.config.batch_size-len(batch_d),
                                                                               len(self.toProcessFrames)))]
            if len(batch_d) > 0:
                batch_lst.append(batch_d)


        #batch = self.compute_batch(batch_lst[0])
        pool = ThreadPool(self.config.num_cpus)
        batch = pool.map(self.compute_batch, batch_lst)
        pool.close()
        pool.join()
        # batch = Parallel(n_jobs=config.num_cpus, verbose=1)(map(delayed(trainData.get_batch), range(2)))
        return batch

    def compute_edge(self,batch_id):

        (sid, src_fid, fid) = batch_id
        locs = self.all_tracks[(sid, src_fid)][fid]
        node_feat = np.zeros((locs.shape[0], (locs.shape[0]), 6))
        counter = 0
        centroids = np.zeros((locs.shape[0], 2)).astype(float)
        centroids[:, 0] = locs[:, 0] + (locs[:, 2]).astype(float) / 2.0
        centroids[:, 1] = locs[:, 1] + (locs[:, 3]).astype(float) / 2.0
        for i in range(locs.shape[0]):
            for j in range(i + 1, locs.shape[0]):
                node_feat[i, j, 0:2] = abs(centroids[i, 0:2] - centroids[j,  0:2])
                node_feat[i, j,  2] = abs(np.sum(centroids[i, 0:2] - centroids[j, 0:2]))
                node_feat[i, j,  3] = np.sqrt(np.sum(np.square(centroids[i, 0:2] - centroids[j, 0:2])))
                node_feat[i, j, 4] = np.arctan((centroids[i, 1] - centroids[j,  1])/(centroids[i, 0] - centroids[j,  0]+self.config.epsilon))
                node_feat[i, j, 5] = np.arctan2(
                    (centroids[i, 1] - centroids[j, 1]), (centroids[i, 0] - centroids[j, 0]))
                node_feat[j, i, :4] = node_feat[i, j, :4]
                node_feat[j, i, 4] = np.arctan(
                    (centroids[j, 1] - centroids[i, 1]) / (centroids[j, 0] - centroids[i, 0]+self.config.epsilon))
                node_feat[j, i, 5] = np.arctan2(
                    (centroids[j, 1] - centroids[i, 1]), (centroids[j, 0] - centroids[i, 0]))
                counter += 1

        return node_feat

    def compute_batch(self, batch_id):
        images, boxes = [], []
        if self.isSRNN:
            edge = []
        actions_lbl, activity_lbl = [], []
        start_time1 = time.time()
        if self.config.training_type == 1:
            all_ids = [ (a,b,b) for a,b in batch_id ]
        else:
            all_ids = [ (a,b,c) for a,b in batch_id for c in range(b-self.config.num_before,b+self.config.num_after+1)]

        if self.config.parallel_loader:
            #out  = dict(Parallel(n_jobs=self.config.num_cpus, verbose=0)(map(delayed(image_read), all_ids)))
            pool = ThreadPool(self.config.num_cpus)
            out = dict(pool.map(image_read, all_ids))
            pool.close()
            pool.join()
        else:
            out = dict([image_read((a,b,c)) for a, b, c in all_ids])
        #print("--- Image read %s seconds ---" % (time.time() - start_time1))
        start_time2 = time.time()
        for i, (sid, src_fid) in enumerate(batch_id):

            imgs, bboxes, edges = [], [], []
            # read image (as temporal list)
            if self.config.training_type == 1:
                bboxes.append(self.all_tracks[(sid, src_fid)][src_fid])
                imgs.append(out[(sid,src_fid,src_fid)])
            else:
                for fid in range(src_fid-self.config.num_before, src_fid+self.config.num_after+1):
                    # img = skimage.io.imread(self.config.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid)) #BGR format
                    # imgs.append(skimage.transform.resize(img, self.config.image_size))
                    imgs.append(out[(sid,src_fid,fid)])
                    bboxes.append(self.all_tracks[(sid, src_fid)][fid])     # add code to convert it to integers if required
                    if self.isSRNN:
                        edges.append(self.compute_edge((sid,src_fid,fid)))
            # images.append(np.stack(out[i]))
            images.append(np.stack(imgs))
            boxes.append(np.stack(bboxes))
            if self.isSRNN:
                edge.append(np.stack(edges))
            actions_lbl.append(np.stack(self.anns[sid][src_fid]['actions']))
            activity_lbl.append(self.anns[sid][src_fid]['group_activity'])

        if self.isSRNN:
            temp = self.__process_batch_based_on_experiment(images, boxes, actions_lbl, activity_lbl, edge)
        else:
            temp = self.__process_batch_based_on_experiment(images,boxes,actions_lbl,activity_lbl)
        #print("--- Rest Data processing %s seconds ---" % (time.time() - start_time2))
        #print("--- Total Data processing %s seconds ---" % (time.time() - start_time1))
        return temp

    def __to_categorical(self,y, nb_classes):
        Y = np.zeros((len(y), nb_classes))
        y = y.astype(np.int)
        for i in range(len(y)):
            # if y[i]>0:
            Y[i, y[i] - 1] = 1.
        return Y.astype(np.bool)

    def __process_batch_based_on_experiment(self, images,boxes,actions_lbl,activity_lbl,edge=None):

        if self.config.feature_type == 1 or self.config.feature_type == 4:

            if self.config.training_type == 1:
                data = zip(images,boxes,actions_lbl)
                out = [crop_images(data[i]) for i in range(len(data))]
                imgLst, action = [], []
                [imgLst.extend(ele[0]) for ele in out]
                [action.extend(ele[1]) for ele in out]
                return np.concatenate(imgLst), self.__to_categorical(np.concatenate(action),self.config.num_actions)
            else:
                start_time1 = time.time()
                if self.isSRNN:
                    data = zip(images, boxes, actions_lbl, edge)
                else:
                    data = zip(images, boxes, actions_lbl)
                out = [crop_images(d) for d in data]

                #print("--- Crop processing %s seconds ---" % (time.time() - start_time1))
                imgLst, action, node_count  = [], [], []
                if self.isSRNN:
                    edgeLst = []
                    edgeMaskLst = []
                for ele in out:
                    imgLst.append(np.stack([ele[0][j] for j in range(len(ele[0]))],axis=1))
                    action.append(np.stack([ele[1][j] for j in range(len(ele[0]))],axis=1))
                    node_count += ele[2]
                    if self.isSRNN:
                        edgeLst.append(np.stack([np.reshape(ele[3][j],(-1,6)) for j in range(len(ele[0]))],axis=1))
                        edgeMaskLst.append(np.stack([ele[4][j] for j in range(len(ele[0]))], axis=2))

                img_data = np.concatenate(imgLst)
                if self.isSRNN:
                    temp_data = np.concatenate(edgeLst)
                    step_size = 3

                    edge_feat = np.zeros((temp_data.shape[0], temp_data.shape[1], temp_data.shape[2]*6),dtype=np.float16)
                    tmp = np.zeros((temp_data.shape[0], temp_data.shape[1], temp_data.shape[2]*2))
                    tmp[:, :, :temp_data.shape[2]] = temp_data
                    tmp[:, step_size:, temp_data.shape[2]:temp_data.shape[2]*2] = \
                        temp_data[:, step_size:, :] - temp_data[:, :-step_size, :]
                    for t in range(temp_data.shape[1]):
                        if t < 0:
                            edge_feat[:, 0, temp_data.shape[2]*2:] = \
                                np.concatenate((tmp[:, 0, :], tmp[:, 1, :]), axis=1)
                        elif t < temp_data.shape[1] - 1:
                            edge_feat[:, t, :] = np.concatenate((tmp[:, t - 1, :], tmp[:, t, :], tmp[:, t + 1, :]),
                                                                 axis=1)
                        elif t == temp_data.shape[1] - 1:
                            edge_feat[:, t, :temp_data.shape[2]*4] = \
                                np.concatenate((tmp[:, t - 1, :], tmp[:, t, :]), axis=1)
                if self.isSRNN:
                    if self.isTraining == False:
                        return img_data, self.__to_categorical(np.concatenate(action), self.config.num_actions), \
                               self.__to_categorical(np.asarray(activity_lbl), self.config.num_activities), \
                               self.__mask_generator(np.asarray(node_count)), edge_feat, self.__emask_generator(edgeMaskLst)

                    elif self.config.training_type == 2 or self.config.training_type == 5:
                        return img_data, self.__to_categorical(np.concatenate(action), self.config.num_actions), \
                               edge_feat, self.__emask_generator(edgeMaskLst)
                    elif self.config.training_type == 3:
                        return img_data, self.__to_categorical(np.asarray(activity_lbl), self.config.num_activities), \
                               self.__mask_generator(np.asarray(node_count)), edge_feat, self.__emask_generator(edgeMaskLst)
                    elif self.config.training_type == 4 or self.config.training_type == 6:
                        return img_data, self.__to_categorical(np.concatenate(action), self.config.num_actions), \
                               self.__to_categorical(np.asarray(activity_lbl), self.config.num_activities), \
                               self.__mask_generator(np.asarray(node_count)), edge_feat, self.__emask_generator(edgeMaskLst)
                    else:
                        raise Exception("Unknown list of training type option")

                if self.isTraining == False:
                    return img_data, self.__to_categorical(np.concatenate(action), self.config.num_actions), \
                           self.__to_categorical(np.asarray(activity_lbl), self.config.num_activities), \
                           self.__mask_generator(np.asarray(node_count))

                elif self.config.training_type == 2 or self.config.training_type == 5:
                    return img_data, self.__to_categorical(np.concatenate(action),self.config.num_actions)
                elif self.config.training_type == 3:
                    return img_data, self.__to_categorical(np.asarray(activity_lbl),self.config.num_activities), \
                           self.__mask_generator(np.asarray(node_count))
                elif self.config.training_type == 4 or self.config.training_type == 6:
                    return img_data, self.__to_categorical(np.concatenate(action),self.config.num_actions),\
                        self.__to_categorical(np.asarray(activity_lbl),self.config.num_activities), \
                           self.__mask_generator(np.asarray(node_count))
                else:
                    raise Exception("Unknown list of training type option")
        return 0

    def __mask_generator(self,arr):
        mask = np.zeros((np.sum(arr),len(arr)),dtype=np.bool)
        for i in range(len(arr)):
            mask[np.sum(arr[:i]):np.sum(arr[:i+1]), i] = True
        return mask

    def __emask_generator(self,maskLst):
        num_ele = sum([m.shape[0] for m in maskLst])
        num_sample = sum([m.shape[1] for m in maskLst])
        mask = np.zeros((num_ele,num_sample,maskLst[0].shape[2],maskLst[0].shape[3]),dtype=np.bool)
        count = 0
        for i in range(len(maskLst)):
            mask[count:count+maskLst[i].shape[0],i*maskLst[i].shape[1]:(i+1)*maskLst[i].shape[1],:,:] = maskLst[i]
            count += maskLst[i].shape[0]
        return mask


def image_read(arg):
    (sid, src_fid, fid) = arg
    config = ExperimentConfig()
    #img = skimage.io.imread(config.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, src_fid))  # BGR format
    img = scipy.misc.imread(config.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, src_fid))  # BGR format
    #img = skimage.transform.resize(img,config.image_size, order=3, mode='constant')
    img = scipy.misc.imresize(img, config.image_size, 'cubic')
    return ((sid,src_fid,fid),img)

def crop_images(arg):
    if len(arg) == 3:
        edge_flag = False
        images, boxes, actions_lbl = arg
    else:
        edge_flag = True
        images, boxes, actions_lbl, edge = arg
    mean = np.array([104., 117., 124.]).astype(np.int16)
    config = ExperimentConfig()
    fimgLst, faction = [], []
    if edge_flag:
        fedgeLst = []
        fedgeMaskLst = []

    point_x = []
    arglst = []
    if config.group_split == 1:
        node_count = [boxes.shape[1]]
    elif config.group_split == 2:
        node_count = [int(boxes.shape[1]/2.0), boxes.shape[1] - int(boxes.shape[1]/2.0)]
    elif config.group_split == 4:
        node_count = [int(boxes.shape[1] / 4.0), int(boxes.shape[1] / 4.0), int(boxes.shape[1] / 4.0),
                      boxes.shape[1] - 3*int(boxes.shape[1] / 4.0)]
    else:
        raise Exception("Wrong Split")
    for i in range(len(images)):            #### Need to update the code for cropping a sequence of images
        img = images[i]
        boxLst = boxes[i]
        actionsLst = actions_lbl
        imgLst, action = [], []
        for j in range(boxLst.shape[0]):

            ## Exact patch (with 5 pixel extra)
            # y1 = np.uint16(round(boxLst[j, 0]*img.shape[0])-5)
            # y2 = min(np.uint16(round(boxLst[j, 2] * img.shape[0])+5),img.shape[0])
            # x1 = np.uint16(round(boxLst[j, 1] * img.shape[1])-5)
            # x2 = min(np.uint16(round(boxLst[j, 3] * img.shape[1])+5),img.shape[1])


            ## square patch (with 5 pixel extra)
            y1 = np.uint16(max(0, round(boxLst[j, 0] * img.shape[0])))
            y2 = min(np.uint16(round(boxLst[j, 2] * img.shape[0])), img.shape[0])
            x1 = np.uint16(max(0, round(boxLst[j, 1] * img.shape[1])))
            x2 = min(np.uint16(round(boxLst[j, 3] * img.shape[1])), img.shape[1])
            x_mid = round(np.float32(x1 + x2) / 2.0)
            y_mid = round(np.float32(y1 + y2) / 2.0)
            if (y2 - y1) > (x2 - x1):
                x1 = np.uint16(max(0, x_mid - (y_mid - y1)))
                x2 = x1 + (y2 - y1)
                if x2 > img.shape[1]:
                    x2 = img.shape[1]
                    x1 = x2 - (y2 - y1)
            elif (x2 - x1) > (y2 - y1):
                y1 = np.uint16(max(0, y_mid - (x_mid - x1)))
                y2 = y1 + (x2 - x1)
                if y2 > img.shape[0]:
                    y2 = img.shape[0]
                    y1 = y2 - (x2 - x1)

            if i == (len(images)/2 - 1):
                point_x.append(x_mid)
            arglst.append((img,y1,y2,x1,x2))

            #patch = np.float32(scipy.misc.imresize(img[y1:y2, x1:x2, :], config.alexnet_size, 'cubic'))
            #patch = np.float32(skimage.transform.resize(img[y1:y2, x1:x2, :], config.alexnet_size))

            # if self.config.data_augementation:
            # add code for data augmentation if required

            #patch -= mean
            #patch /= 127

    pool = ThreadPool(config.num_cpus)
    out = pool.map(c_r, arglst)
    pool.close()
    pool.join()
    out.reverse()
    sort_indx = [i[0] for i in sorted(enumerate(point_x), key=lambda x: x[1])]

    for i in range(len(images)):
        boxLst = boxes[i]
        actionsLst = actions_lbl
        imgLst, action = [], []
        for j in range(boxLst.shape[0]):
            #imgLst.append(patch)
            imgLst.append(out.pop())
            action.append(actionsLst[j])
        fimgLst.append(np.stack(imgLst)[sort_indx,:,:,:])
        faction.append(np.stack(action)[sort_indx,])
        if edge_flag:
            fedgeMaskLst.append(process_edge(boxLst[sort_indx, :]))
            fedgeLst.append(edge[i][sort_indx, :, :][:, sort_indx, :])
    # print len(fimgLst)
    # raise Exceptioni("Done")
    if edge_flag:
        return  fimgLst, faction, node_count, fedgeLst, fedgeMaskLst
    return fimgLst, faction, node_count

def c_r(args):
    (img, y1, y2, x1, x2) = args
    config = ExperimentConfig()
    mean = np.array([104., 117., 124.]).astype(np.int16)
    patch = np.int16(scipy.misc.imresize(img[y1:y2, x1:x2, :], config.alexnet_size, 'cubic'))
    patch -= mean
    if config.feature_type == 4:
        patch = patch[:,:,[2,1,0]]
    return patch

def process_edge(boxes):
    config = ExperimentConfig()
    ## This piece code further processes edges
    centroids = np.zeros((boxes.shape[0], 2)).astype(float)
    centroids[:, 0] = boxes[:, 0] + (boxes[:, 2]).astype(float) / 2.0
    centroids[:, 1] = boxes[:, 1] + (boxes[:, 3]).astype(float) / 2.0
    mask = []
    for i in range(boxes.shape[0]):
        l1 = np.asarray(centroids[:, 0] <= centroids[i, 0])
        l1[i] = False
        l2 = np.asarray(centroids[:, 0] > centroids[i, 0])
        l2[i] = False
        l3 = np.asarray(centroids[:, 1] <= centroids[i, 1])
        l3[i] = False
        l4 = np.asarray(centroids[:, 1] > centroids[i, 1])
        l4[i] = False
        l5 = np.all(np.stack((l1, l3), axis=1), axis=1)
        l6 = np.all(np.stack((l1, l4), axis=1), axis=1)
        l7 = np.all(np.stack((l2, l4), axis=1), axis=1)
        l8 = np.all(np.stack((l2, l3), axis=1), axis=1)
        mask.append(np.stack([l1,l2,l3,l4,l5,l6,l7,l8],axis=1))

    mask = np.stack(mask,axis=0)
    ## Process mask to have sliding window
    m1 = np.transpose(mask, axes=[1, 0, 2])
    m2 = np.zeros((m1.shape[0] * m1.shape[1], config.max_nodes, m1.shape[2]), dtype=np.bool)
    for i in range(m1.shape[1]):
        m2[i * m1.shape[1]:(i + 1) * m1.shape[1], i, :] = m1[:, i, :]
    return m2
