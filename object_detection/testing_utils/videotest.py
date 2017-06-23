""" A class for testing a SSD model on a video file or webcam """

import cv2

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

from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer

import sys
sys.path.append("..")

from kobuki import Kobuki

CENTER_THRESHOLD = 30

class VideoTest(object):
    """ Class for testing a trained SSD model on a video file and show the
        result in a window. Class is designed so that one VideoTest object
        can be created for a model, and the same object can then be used on
        multiple videos and webcams.

        Arguments:
            class_names: A list of strings, each containing the name of a class.
                         The first name should be that of the background class
                         which is not used.

            model:       An SSD model. It should already be trained for
                         images similar to the video to test on.

            input_shape: The shape that the model expects for its input,
                         as a tuple, for example (300, 300, 3)

            bbox_util:   An instance of the BBoxUtility class in ssd_utils.py
                         The BBoxUtility needs to be instantiated with
                         the same number of classes as the length of
                         class_names.

    """

    def __init__(self, category_index, num_classes, detection_graph, input_shape):
        #self.class_names = class_names
        #self.num_classes = len(class_names)
        self.category_index = category_index
        self.num_classes = num_classes
        self.detection_graph = detection_graph
        self.input_shape = input_shape
        self.kobuki = Kobuki()

        # Create unique and somewhat visually distinguishable bright
        # colors for the different classes.
        self.class_colors = []
        for i in range(0, self.num_classes):
            # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col)

    def predict(self, image):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.array(image).reshape(
            (self.input_shape[0], self.input_shape[1], 3)).astype(np.uint8)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        with tf.Session(graph=self.detection_graph) as sess:
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            #print type(boxes), boxes[0]
            #print type(scores), scores[0]
            #print type(classes), classes[0]
            #print num_detections

            return (boxes, scores, classes, num_detections)

    def run(self, video_path = 0, start_frame = 0, conf_thresh = 0.6, target_class_label = None):
        """ Runs the test on a video (or webcam)

        # Arguments
        video_path: A file path to a video to be tested on. Can also be a number,
                    in which case the webcam with the same number (i.e. 0) is
                    used instead

        start_frame: The number of the first frame of the video to be processed
                     by the network.

        conf_thresh: Threshold of confidence. Any boxes with lower confidence
                     are not visualized.

        """

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))

        # Compute aspect ratio of video
        #vidw = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        #vidh = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vidar = vidw/vidh

        # Skip frames until reaching start_frame
        if start_frame > 0:
            #vid.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_frame)
            vid.set(cv2.CAP_PROP_POS_MSEC, start_frame)

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        pmin = None
        pmax = None

        while True:
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                return

            im_size = (self.input_shape[0], self.input_shape[1])
            resized = cv2.resize(orig_image, im_size)

            # Reshape to original aspect ratio for later visualization
            # The resized version is used, to visualize what kind of resolution
            # the network has to work with.
            to_draw = cv2.resize(resized, (int(self.input_shape[0]*vidar), self.input_shape[1]))

            # predict
            (boxes, scores, classes, num_detections) = self.predict(image=resized)

            if num_detections > 0:
                # Interpret output, only one frame is used
                det_label = classes[0]
                det_conf = scores[0]
                # see utils/visualization_utils.py visualize_boxes_and_labels_on_image_array()
                det_ymin = boxes[0][:, 0]
                det_xmin = boxes[0][:, 1]
                det_ymax = boxes[0][:, 2]
                det_xmax = boxes[0][:, 3]

                if target_class_label is None:
                    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
                else:
                    top_indices = [i for i, (conf, label) in enumerate(zip(det_conf, det_label)) if conf >= conf_thresh and label == target_class_label]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                # top_xmin = det_xmin[top_indices]
                # top_ymin = det_ymin[top_indices]
                # top_xmax = det_xmax[top_indices]
                # top_ymax = det_ymax[top_indices]
                top_min = np.array([det_xmin[top_indices], det_ymin[top_indices]]).T
                top_max = np.array([det_xmax[top_indices], det_ymax[top_indices]]).T

                target_index = None
                # select target
                if target_class_label is not None:
                    if pmin is None:
                        # target (at first, get argmax conf)
                        target_index = np.argmax(top_conf)
                    else:
                        # target (secondly, search nearly previous xmin/ymax)
                        loss = 999999
                        for i in range(top_conf.shape[0]):
                            l = np.linalg.norm(top_min[i] - pmin) + np.linalg.norm(top_max[i] - pmax)
                            if l < loss:
                                target_index = i
                                loss = l

                # chase target
                if target_index is not None:
                    pmin = top_min[target_index]
                    pmax = top_max[target_index]

                    #judge center by x
                    xwidth = to_draw.shape[1]
                    xmin = int(round(top_min[target_index][0] * to_draw.shape[1]))
                    ymin = int(round(top_min[target_index][1] * to_draw.shape[0]))
                    xmax = int(round(top_max[target_index][0] * to_draw.shape[1]))
                    ymax = int(round(top_max[target_index][1] * to_draw.shape[0]))
                    loss = (xmax + xmin - xwidth) / 2
                    if np.absolute(loss) < CENTER_THRESHOLD:
                        # target is center
                        self.kobuki.go_forward()
                    elif loss < 0:
                        # target is left
                        self.kobuki.turn_right()
                    else:
                        # target is right
                        self.kobuki.turn_left()
                else:
                    pmin = None

                # draw all rectangle
                for i in range(top_conf.shape[0]):
                    # xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                    # ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                    # xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                    # ymax = int(round(top_ymax[i] * to_draw.shape[0]))
                    xmin = int(round(top_min[i][0] * to_draw.shape[1]))
                    ymin = int(round(top_min[i][1] * to_draw.shape[0]))
                    xmax = int(round(top_max[i][0] * to_draw.shape[1]))
                    ymax = int(round(top_max[i][1] * to_draw.shape[0]))

                    # Draw the box on top of the to_draw image
                    class_num = int(top_label_indices[i])
                    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax),
                                  self.class_colors[class_num], 2 if i is not target_index else 10)
                    #text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])
                    text = self.category_index[class_num]['name'] + " " + ('%.2f' % top_conf[i])

                    text_top = (xmin, ymin-10)
                    text_bot = (xmin + 80, ymin + 5)
                    text_pos = (xmin + 5, ymin)
                    cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)
                    cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

            # Calculate FPS
            # This computes FPS for everything, not just the model's execution
            # which may or may not be what you want
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            # Draw FPS in top left corner
            cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

            cv2.imshow("SSD result", to_draw)
            cv2.waitKey(10)
