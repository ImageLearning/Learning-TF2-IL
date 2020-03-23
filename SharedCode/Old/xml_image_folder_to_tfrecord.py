from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import os
import cv2
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from enum import Enum
from random import shuffle
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


#Image Max Width
MaxWidth = 1024
#Image Max Height
MaxHeight = 1024
# This many images must have a tag for it to be included
ThresholdForInclusion = 30
Training_Safety_Net = 20
# What percentage of images should we use for trian VS test (0.8 = 80%)
Target_Train_Percent = 0.80

flags = tf.app.flags
flags.DEFINE_string('img_dir', 'Data/DataPreperation/octocats', 'Parent folder of the image directories')
flags.DEFINE_string('test_dir', 'Data/DataPreperation/octocats/test', 'Parent folder of the image directories')
flags.DEFINE_string('train_dir', 'Data/DataPreperation/octocats/train', 'Parent folder of the image directories')
# You probably want         Tensorflow/models/research/object_detection/test.record
flags.DEFINE_string('output_path_test', 'Data/TensorflowRecords/test.record', 'Path to output test TFRecord')
# You probably want         Tensorflow/models/research/object_detection/train.record
flags.DEFINE_string('output_path_train', 'Data/TensorflowRecords/train.record', 'Path to output train TFRecord')
# You probably want         Tensorflow/models/research/object_detection/training/labelmap.pbtxt
flags.DEFINE_string('output_path_labelmap', 'Data/TensorflowRecords/labelmap.pbtxt', 'Path to output labelmap.pbtxt')
flags.DEFINE_string('config_file_path', 'Data/TensorflowConfig/faster_rcnn_inception_v2_pets.template', 'Path to the template config file')

flags.DEFINE_string('fine_tune_checkpoint', 'Data/TensorflowDatasets/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt', 'Path to the checkpoint we are building off of')

flags.DEFINE_string('resize_images', 'False', "True = Resize images before packing TFRecord")

FLAGS = flags.FLAGS

##########################################################
## --- Maybe don't change things below here?
##########################################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

Resize = False

# All initial Records found
MapOfRecords = dict()
# All initial Labels found
AllLabels = dict()
# Count of each label found
AllLabelsCount = dict()
# list of all labels found
LabelList = list()

# Only the labels we are going to use (enough images to meet the threshold)
UsableLabelMap = dict()

Current_Label_Index = 1

# Things to Print on Exit
Print_NumClasses = 0
Print_NumTestImages = 0
Print_NumTrainImages = 0

dir_path = os.getcwd()

CONFIG_CLASSES = "%CLASSES%"
CONFIG_MAX_DIMENSION = "%MAX_DIMENSION%"
CONFIG_TRAIN_INPUT_PATH = "%TRAIN_INPUT_PATH%"
CONFIG_TEST_INPUT_PATH = "%TEST_INPUT_PATH%"
CONFIG_TRAIN_LABEL_MAP = "%TRAIN_LABELMAP%"
CONFIG_TEST_LABEL_MAP = "%TEST_LABELMAP%"
CONFIG_EVAL_NUM_TEST_IMG = "$EVAL_NUM_EXAMPLES$"
CONFIG_FINE_TUNE_CHECKPOINT = "$FINE_TUNE_CHECKPOINT$"


#How will this image be used? TBD, Dedicated Test image, Dedicated Training image
class ImgUsage(Enum):
    TBD = 1
    TRAIN = 2
    TEST = 3

# Used to append _resized onto image file names
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

# A record full of bounding rectangles, used after we determine which images we wil train
class PackRecord:
    def __init__(self):
        self.filename = "TBD"
        self.width = 0
        self.height = 0
        self.usage = ImgUsage.TBD
        self.HasBeenResized = False
        self.BoundingRectangles = []
    def __init__(self, pfilename:str,pwidth:int,pheight:int,pusage:ImgUsage,pHasBeenResized:bool):
        self.filename = pfilename
        self.width = pwidth
        self.height = pheight
        self.usage = pusage
        self.HasBeenResized = False
        self.BoundingRectangles = []

# Bounding Rectangles all within a single image (packrecord)
class BoundingRectangle:
    def __init__(self, pparent:PackRecord,pxmin:int,pxmax:int,pymin:int,pymax:int,plabel:str,phasbeenresized:bool):
        self.parent = pparent
        self.xmin = pxmin
        self.xmax = pxmax
        self.ymin = pymin
        self.ymax = pymax
        self.label = plabel
        self.HasBeenResized = phasbeenresized

# Temporary Data holder, used to help determine which images we will use to train
class ImageLocationRecord:
    def __init__(self):
        self.filename = "TBD"
        self.width = 0
        self.height = 0
        self.label = "TBD"
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.usage = ImgUsage.TBD
        self.HasBeenResized = False


def class_text_to_int(label : str)->int:
    global UsableLabelMap
    if label in UsableLabelMap:
        return UsableLabelMap[label]
    else:
        print(bcolors.FAIL + label + " : is missing" + bcolors.ENDC)

def validate_or_add_label(label : str):
    global AllLabels
    global AllLabelsCount
    global Current_Label_Index
    if label in AllLabels:
        AllLabelsCount[label] += 1
    else:    
        LabelList.append(label)
        AllLabels[label] = Current_Label_Index
        AllLabelsCount[label] = 1
        Current_Label_Index += 1

def check_resize_image(record : PackRecord):
    global Resize
    shouldresize = False
    if (record.width > MaxWidth):
        shouldresize = True 
    if (record.height > MaxHeight):
        shouldresize = True 

    if(shouldresize and Resize):
        WidthFactor = MaxWidth / record.width * 1.0
        HeightFactor = MaxHeight / record.height * 1.0

        ResizeFactor = 1.0
        if(WidthFactor > HeightFactor):
            ResizeFactor = HeightFactor 
        else:
            ResizeFactor = WidthFactor


        #ok Now Resize
        image = cv2.imread(record.filename)
        if(image is None):
            print(bcolors.FAIL + "Image Missing " + str(record.filename) + bcolors.ENDC )
            quit()

        resized = cv2.resize(image,None,fx=ResizeFactor, fy=ResizeFactor, interpolation=cv2.INTER_AREA)
        record.filename = rreplace( record.filename,".", "_resized.",1)

        cv2.imwrite(record.filename,resized)

        
        print(bcolors.OKGREEN + "Re-size to : " + record.filename + " by a factor of " + str(ResizeFactor) + "" + bcolors.ENDC)

        #ok now change all the numbers
        record.width *= ResizeFactor
        record.height *= ResizeFactor
        record.HasBeenResized = True
        for br in record.BoundingRectangles:
            br.xmin *= ResizeFactor
            br.xmax *= ResizeFactor
            br.ymin *= ResizeFactor
            br.ymax *= ResizeFactor
    else:
        print(bcolors.OKGREEN + "Not Re-sizeing, but should : " + record.filename + bcolors.ENDC)


def Load_XML(img_dir:str,train_dir:str,test_dir:str):
    #xml_df = []
    for root,directories,files in os.walk(img_dir):
        for folder in directories:
            image_path = os.path.join(root, folder)
            usage = ImgUsage.TBD
            if(image_path == train_dir):
                usage = ImgUsage.TRAIN
            elif(image_path == test_dir):
                usage = ImgUsage.TEST
            #xml_df += ( 
            xml_to_imagerecord(image_path, usage) 
            #)
            #xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
            print( bcolors.OKBLUE + 'Successfully loaded all xml records from :' + str(image_path) + " " + bcolors.ENDC )
    #return xml_df

def xml_to_imagerecord(path, usage:ImgUsage):
    global MapOfRecords
    #xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = ImageLocationRecord()
            value.filename = os.path.join(path, root.find('filename').text)
            value.width = int(root.find('size')[0].text)
            value.height = int(root.find('size')[1].text)
            value.label = member[0].text
            value.xmin = int(member[4][0].text)
            value.ymin = int(member[4][1].text)
            value.xmax = int(member[4][2].text)
            value.ymax = int(member[4][3].text)
            value.usage = usage
            #Did we learn any new lables?
            validate_or_add_label(value.label)

            #xml_list.append(value)

            if value.label in MapOfRecords: 
                MapOfRecords[value.label].append(value) 
            else: 
                MapOfRecords[value.label]=[value] 

    #return xml_list


def ValidateUsableLabels()->list:
    global ThresholdForInclusion
    global AllLabelsCount
    print(bcolors.OKBLUE + "discarding images with less than : " + str(ThresholdForInclusion) + " images" + bcolors.ENDC)
    new_records = dict()
    usable_label = list()
    for key, value in sorted(AllLabelsCount.items()): 
        if(value > ThresholdForInclusion):
            usable_label.append(key)
            if value in new_records: 
                new_records[value].append(key) 
            else: 
                new_records[value]=[key] 
        else:
            print(bcolors.OKGREEN + key + " with " + str(value) + " records was not included" + bcolors.ENDC)

    print("---------")
    imagecount = 0
    NewIndex = 0
    for value in usable_label:
        temp_img_count = AllLabelsCount[value]
        NewIndex += 1
        # This is where we create the UsableLabels map
        UsableLabelMap[value] = NewIndex
        print(bcolors.OKBLUE + "" + value + " has : " + str(temp_img_count) + " entries" + bcolors.ENDC)
        imagecount += temp_img_count

    print(bcolors.HEADER + "---------")
    print("We have : " + str(imagecount) + " labeled bounding boxes" +  bcolors.ENDC)

    if(imagecount == 0):
        assert("It's all gone wrong")

    return usable_label

def PrintLabelMap(usable_label:list, labelmap_dir)->None:
    f = open(labelmap_dir, "w")   
    for value in usable_label:
        f.write("item {\n")
        f.write("  id: " + str( class_text_to_int(value) ) + "\n")
        f.write("  name: '" + value + "'\n")
        f.write("}\n")
        f.write(" \n")

    f.close()


def FindUsableImges(usable_labels:list)->[list,list]:
    global Target_Train_Percent
    global Print_NumClasses 
    global Print_NumTestImages 
    global Print_NumTrainImages 

    recordlist = list()
    trainingrecordlist = list()
    testrecordlist = list()

    imagecount_withDuplicates = 0
    for value in usable_labels:
        imagecount_withDuplicates += AllLabelsCount[value]
        recordlist += MapOfRecords[value]
        Print_NumClasses += 1

    imagecount = 0
    temp = dict()
    for value in recordlist:
        if value.filename in temp: 
            tempPackRecord = temp[value.filename]
            tempPackRecord.BoundingRectangles.append(BoundingRectangle(tempPackRecord,value.xmin,value.xmax, value.ymin, value.ymax, value.label, value.HasBeenResized) )
            if(value.usage == ImgUsage.TRAIN):
                tempPackRecord.usage = ImgUsage.TRAIN
            elif(value.usage == ImgUsage.TEST and tempPackRecord.usage != ImgUsage.TRAIN):
                tempPackRecord.usage = ImgUsage.TEST            
        else: 
            temp[value.filename] = PackRecord(value.filename,value.width,value.height,value.usage,value.HasBeenResized)
            tempPackRecord = temp[value.filename]
            tempPackRecord.BoundingRectangles.append(BoundingRectangle(tempPackRecord,value.xmin,value.xmax, value.ymin, value.ymax, value.label, value.HasBeenResized) )
            if(value.usage == ImgUsage.TRAIN):
                tempPackRecord.usage = ImgUsage.TRAIN
            elif(value.usage == ImgUsage.TEST and tempPackRecord.usage != ImgUsage.TRAIN):
                tempPackRecord.usage = ImgUsage.TEST  
            imagecount += 1

    

    target_train_records = int(Target_Train_Percent * imagecount)
    target_test_records = imagecount - target_train_records
    #clean this up later, probably only need the global variables
    Print_NumTestImages = target_test_records
    Print_NumTrainImages = target_train_records
    print("+--- Out of " + str(imagecount) + " images...")
    print(" Training with : " + str(target_train_records))
    print(" Testing with : " + str(target_test_records))

    nodupes_recordlist = list()

    for value in temp.values():
        nodupes_recordlist.append(value)

    # Now lets remove the images we are determine to train/test with
    for value in nodupes_recordlist:
        if(value.usage == ImgUsage.TRAIN):
            trainingrecordlist.append(value)
            target_train_records -= 1
        elif(value.usage == ImgUsage.TEST):
            testrecordlist.append(value)
    
    for value in trainingrecordlist:
        nodupes_recordlist.remove(value)
    for value in testrecordlist:
        nodupes_recordlist.remove(value)

    shuffle(nodupes_recordlist)

    # Due to Random, in theory we could have 0 test items from a single class
    # Probably should go through and validate that at some point... 
    # In a perfect world, the user would put 80% of images in the test directory, and nothing would be needed here
    for value in nodupes_recordlist:
        if(target_train_records > 0):
            value.usage = ImgUsage.TRAIN
            trainingrecordlist.append(value)
            target_train_records -= 1
        else:
            testrecordlist.append(value)

    return trainingrecordlist,testrecordlist



def WriteDataset(records,output_path):    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

    for value in records:

        # Add Re-Size Right Here
        check_resize_image(value)

        with tf.gfile.GFile(value.filename, 'rb') as fid:
            encoded_img = fid.read()

        encoded_img_io = io.BytesIO(encoded_img)
        image = Image.open(encoded_img_io)
        width, height = image.size
        width = int(value.width)
        height = int(value.height)
        filename = value.filename.encode('utf8')
        image_format = b'jpg'
        if(value.filename.lower().endswith('.png')):
            image_format = b'png'    

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for boundingbox in value.BoundingRectangles:
            xmins.append(float(boundingbox.xmin) / float(width))
            xmaxs.append(float(boundingbox.xmax) / float(width))
            ymins.append(float(boundingbox.ymin) / float(height))
            ymaxs.append(float(boundingbox.ymax) / float(height))
            classes_text.append(boundingbox.label.encode('utf8'))
            classes.append( int(class_text_to_int(boundingbox.label) ) )


        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height': dataset_util.int64_feature( int(height) ),
                    'image/width': dataset_util.int64_feature( int(width) ),
                    'image/filename': dataset_util.bytes_feature(filename),
                    'image/source_id': dataset_util.bytes_feature(filename),
                    'image/encoded': dataset_util.bytes_feature(encoded_img),
                    'image/format': dataset_util.bytes_feature(image_format),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                    }
                )
            )
        
        writer.write(tf_example.SerializeToString())
        print( str(len(value.BoundingRectangles)) + " labels from:" + str(filename) + " height:"  +str(height) + " width:" + str(width) + " Saved")


    writer.close()
    
    print('Successfully created the TFRecords: {}'.format(output_path))


def WriteConfigFile(Configfilename:str, LabelMap:str, output_path_test:str, output_path_train:str, num_test_images:int, local_fine_tune_checkpoint:str):
    global CONFIG_CLASSES
    global Print_NumClasses

    global CONFIG_MAX_DIMENSION 
    global MaxHeight
    global MaxWidth

    global CONFIG_TRAIN_INPUT_PATH 
    global CONFIG_TEST_INPUT_PATH 

    global CONFIG_TRAIN_LABEL_MAP
    global CONFIG_TEST_LABEL_MAP

    global CONFIG_EVAL_NUM_TEST_IMG

    global CONFIG_FINE_TUNE_CHECKPOINT

    with open(Configfilename, 'r') as filehandle:
        file_string = filehandle.read()

    # Replace the Config Class placeholder text with our actual data
    file_string = file_string.replace(CONFIG_CLASSES, str(Print_NumClasses))
    
    #Max Image Size
    if(MaxHeight > MaxWidth):
        file_string = file_string.replace(CONFIG_MAX_DIMENSION, str(MaxHeight))
    else:
        file_string = file_string.replace(CONFIG_MAX_DIMENSION, str(MaxWidth))
  
    file_string = file_string.replace(CONFIG_TEST_INPUT_PATH, output_path_test.replace('\\',"/") )
    file_string = file_string.replace(CONFIG_TRAIN_INPUT_PATH, output_path_train.replace('\\',"/") )

    file_string = file_string.replace(CONFIG_TRAIN_LABEL_MAP, LabelMap.replace('\\',"/") )
    file_string = file_string.replace(CONFIG_TEST_LABEL_MAP, LabelMap.replace('\\',"/") )

    file_string = file_string.replace(CONFIG_EVAL_NUM_TEST_IMG, str(num_test_images))

    
    file_string = file_string.replace(CONFIG_FINE_TUNE_CHECKPOINT, local_fine_tune_checkpoint.replace('\\',"/") )
 
    save_file_name = rreplace( Configfilename,".template", ".config",1)

    with open(save_file_name,"w") as filehandle:
        filehandle.write(file_string)
        filehandle.close()


def main(_):
    img_dir = os.path.join(dir_path, FLAGS.img_dir)
    train_dir = os.path.join(dir_path, FLAGS.train_dir)
    test_dir = os.path.join(dir_path, FLAGS.test_dir)
    labelmap_dir = os.path.join(dir_path, FLAGS.output_path_labelmap)
    config_file_path = os.path.join(dir_path, FLAGS.config_file_path)
    Resize = bool(FLAGS.resize_images)
    finetunecheckpoint = os.path.join( dir_path, FLAGS.fine_tune_checkpoint )

    Load_XML(img_dir,train_dir,test_dir)

    # Figure out which images we are going to use
    usable_label = ValidateUsableLabels()
    
    if(len(usable_label) == 0):
        print("No Usable Images Found")
    else:
        PrintLabelMap(usable_label,labelmap_dir)  

    train_records,test_records = FindUsableImges(usable_label) 
    
    WriteDataset(test_records,FLAGS.output_path_test)

    WriteDataset(train_records,FLAGS.output_path_train)

    
    print("Number of Clases to Train: " + str(Print_NumClasses))
    print("Number of Test Images: " + str(Print_NumTestImages))
    print("Number of Training Images: " + str(Print_NumTrainImages))



    WriteConfigFile(config_file_path, labelmap_dir, os.path.join(dir_path, FLAGS.output_path_test),os.path.join(dir_path, FLAGS.output_path_train), Print_NumTestImages, finetunecheckpoint)




if __name__ == '__main__':
    tf.app.run()

# python legacy_train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_atrous_pets.config 
# python tensorboard --logdir=../Data/TensorflowOutput
# python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_resnet_v2_atrous_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
