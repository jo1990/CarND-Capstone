from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime
import rospy
import yaml


class TLClassifier(object):
    def __init__(self):

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.safe_load(config_string)

        #https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/
        model_path = './light_classification/model/tl_classifier_tf.pb'

        self.model = tf.Graph()
        with self.model.as_default():
            graph_defintion = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_defintion.ParseFromString(fid.read())
                tf.import_graph_def(graph_defintion, name='')

            self.model_num_detections = self.model.get_tensor_by_name('num_detections:0')
            
            self.model_image_tensor = self.model.get_tensor_by_name('image_tensor:0')
            
            self.model_detection_boxes = self.model.get_tensor_by_name('detection_boxes:0')
            
            self.model_detection_scores = self.model.get_tensor_by_name('detection_scores:0')
            
            self.model_detection_classes = self.model.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.model)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: traffic light color ID (specified in styx_msgs/TrafficLight)

        """
        with self.model.as_default():

            img_np = np.expand_dims(image, axis=0)

            (detected_boxes,
             detected_scores,
             detected_classes,
             detected_num_detections) = self.sess.run([self.model_detection_boxes,
                                                    self.model_detection_scores,
                                                    self.model_detection_classes,
                                                    self.model_num_detections],
                                         feed_dict={self.model_image_tensor:
                                                        img_np})

        detected_boxes = np.squeeze(detected_boxes)
        detected_scores = np.squeeze(detected_scores)
        detected_classes = np.squeeze(detected_classes).astype(np.int32)

        # only return classifictions which have a
        # minimum probability distribution of 0.6
        if detected_scores[0] > .6:
            if detected_classes[0] == 1:
                rospy.loginfo("TL-Classifier :: Green Light Detected")
                return TrafficLight.GREEN
            elif detected_classes[0] == 2:
                rospy.loginfo("TL-Classifier :: Red Light Detected")
                return TrafficLight.RED
            elif rdetected_classes[0] == 3:
                rospy.loginfo("TL-Classifier :: Yellow Light Detected")
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
