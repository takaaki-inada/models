import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image

import sys
sys.path.append("object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

from videotest import VideoTest

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
#BASE_PATH = '../'
BASE_PATH = 'object_detection/'
PATH_TO_CKPT = BASE_PATH + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = BASE_PATH + 'data/mscoco_label_map.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print category_index

input_shape = (300,300,3)

vid_test = VideoTest(category_index, NUM_CLASSES, detection_graph, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
#vid_test.run('path/to/your/video.mkv')

#vid_test.run('/home/vagrant/Videos/ILSVRC2015_train_00755001.mp4', conf_thresh = 0.5, target_class_label = 1) # 1:person

#vid_test.run(conf_thresh = 0.5, target_class_label = None)
vid_test.run(conf_thresh = 0.5, target_class_label = 1) # 1:person
