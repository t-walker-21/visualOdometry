from feature_detection import FeatureExtractor
import cv2
import argparse

arg_parser = argparse.ArgumentParser()

# add arguments to parser
arg_parser.add_argument('-i', '--image', required=True, help='Image')
arg_parser.add_argument('-j', '--image_2', required=True, help='Image_2')



args = vars(arg_parser.parse_args())

ext = FeatureExtractor()

img_1 = args['image']
img_2 = args['image_2']

image1 = cv2.imread(img_1)
image2 = cv2.imread(img_2)

ext.feature_matcher(image1, image2, features="surf")