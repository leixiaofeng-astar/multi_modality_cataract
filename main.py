# -*- coding: utf-8 -*-
"""
This module is for cataract model validation.
Commands:


python3 main.py --image_type diffuse --input image_file

-- TODO: support in future
python3 main.py --input INPUT_FILEPATH --label LABEL_FILE --output outputs --threshold float_value
python3 main.py --input images --label ground_truth_sample.csv --output outputs --threshold 0.013259

"""

import time
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import cv2
import torch.nn as nn
import argparse
import csv
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.preprocessing import label_binarize
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

from own_models import *


prj_path = './'
error_code = 0
# MODEL_NAME = 'referable_0.48_16July_Resnet50.dat'
# XGB_MODEL_NAME = 'any048_class_model4ResNet18_xgb131-AUC9703.pth'
model_index = 0
models = ['Resnet18', 'Resnet50', 'Densenet112', 'inres']


def nparray2tensor(x, cuda=True):
    normalize = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if x.shape[-1] == 1:
        x = np.tile(x, [1, 1, 3])

    # ToTensor() expects [h, w, c], so transpose first
    x2 = normalize(transforms.ToPILImage()(x.copy()))
    return x2

def feat_extract(model, img):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    img_num = img.shape[0]
    img_x = img.transpose(0,3,2,1)
    # img_x = img.transpose(0, 3, 1, 2)
    for i in range(img_num):
        img_tmp = img[i,:,:,:]
        img_tmp = img_tmp.astype(np.uint8)

        # x = img[:, ::-1, :, :]

        img_x[i,:,:,:] = nparray2tensor(img_tmp, True)

    img_x = torch.from_numpy(img_x)
    if torch.cuda.is_available():
        model = model.to(device)
        model.eval()

    model.eval()
    batch_size = 16
    feature_test = []
    for i in range(0, img_num, batch_size):
        start_idx = i
        end_idx = i + batch_size
        if end_idx >= img_num:
            end_idx = img_num

        input_img = img_x[start_idx:end_idx, :, :, :]
        input_img = input_img.to(device)
        feat, out = model(input_img)

        feature_test.append(feat.detach().cpu().numpy())

    feat = np.vstack(feature_test)
    # feat = np.reshape(feature_test, (feature_test.shape[0], -1))

    return feat

# auc_value = plot_auc(y, y_pred, savename)
def plot_auc(y_test, y_pred, save_name):
    cls_max = int(np.max(y_test))
    label_name = 'visual_impairment_corrected'
    y_mult = label_binarize(y_test, classes=range(cls_max + 1))

    for c in range(cls_max):
        if cls_max > 1:
            y_truth = y_mult[:, c + 1]
        else:
            y_truth = y_test

        fpr, tpr, thresholds = roc_curve(y_truth, y_pred[:, c + 1])
        # fpr, tpr, thresholds = roc_curve(y_truth, y_pred)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s ROC - class %d' % (label_name, c + 1))
        plt.savefig("pre_new_model.jpg", format='JPEG')
        plt.legend(loc="lower right")

        plt.show()
    return roc_auc


def regression_plot(y, y_pred_class):
    plt.scatter(y, y_pred_class, s=0.2)
    plt.plot([-2, 2], [-2, 2], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 2.0])
    # plt.ylim([0.0, 2.05])
    plt.xlabel('Ground truth')
    plt.ylabel('Estimated')
    plt.title('Regression plot')
    plt.legend(loc="lower right")
    plt.savefig("../fig/pre_new_model.jpg", format='JPEG')
    plt.show()
    return


class Fundus_Dataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
                Note: we don't use it in this project, we keep it for future usage
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
            target_transform: optional transform to label image.
        """
        imgs = []
        if image_list_file is not None:
            for line in image_list_file:
                img = line[0]
                mask = line[1]
                imgs.append((img, mask))
        else:
            # insert with dummy label 0
            for each in os.listdir(data_dir):
                if each.startswith(".") or each.endswith(".csv"):
                    continue
                imgs.append((each, 0))

        #imgs = make_dataset(root)
        self.data_path = data_dir
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        x_path, label = self.imgs[index]

        # img_x is array
        file_with_path = os.path.join(self.data_path, x_path)
        imgs = readImageFile(file_with_path)
        head, tail = os.path.split(file_with_path)
        self.filename = tail
        img_y = int(label)

        img_x = imgs

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)

    def __current_filename__(self):
        return self.filename


def is_img(ext):
    ext = ext.lower()
    if ext == '.jpg' or ext == '.JPG' :
        return True
    elif ext == '.png' or ext == '.PNG':
        return True
    elif ext == '.jpeg' or ext == '.JPEG':
        return True
    elif ext == '.bmp' or ext == '.BMP':
        return True
    elif ext == '.tif' or ext == '.TIF':
        return True
    elif ext == '.tiff' or ext == '.TIFF':
        return True
    elif ext == '.dcm' or ext == '.dicom':
        return True
    elif ext == '.DCM' or ext == '.DICOM':
        return True
    else:
        return False

def read_diffuse_slitlamp_File(FilewithPath):
    # we could repeat a few times in case the network file transder is not done
    img = None
    fileName = os.path.basename(FilewithPath)
    ext_str = os.path.splitext(fileName)[1]

    image_transforms = transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    repeat_time = 3
    if is_img(ext_str):
        # print("Input Image: ", fileName)
        cycle_cnt = repeat_time
        while cycle_cnt>0 and img is None:
            # try one more time in case libpng error: Read Error
            try:
                # img = cv2.imread(FilewithPath, cv2.IMREAD_COLOR)
                # img = cv2.resize(img, (224, 224))
                #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                # import pdb
                # pdb.set_trace()
                img = Image.open(FilewithPath)
                img = image_transforms(img)
            except:
                print("read image error, ignore the file")

            if img is None:
                cycle_cnt = cycle_cnt - 1
                time.sleep(0.02 * (repeat_time-cycle_cnt))

        if img is not None and img.shape[2] == 1:
            # repeat 3 times to make fake RGB images
            img = np.tile(img, [1, 1, 3])

    return img

def readImageFile(FilewithPath):
    # we could repeat a few times in case the network file transder is not done
    img = None
    fileName = os.path.basename(FilewithPath)
    ext_str = os.path.splitext(fileName)[1]
    repeat_time = 3
    if is_img(ext_str):
        # print("Input Image: ", fileName)
        cycle_cnt = repeat_time
        while cycle_cnt>0 and img is None:
            # try one more time in case libpng error: Read Error
            try:
                img = cv2.imread(FilewithPath, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            except:
                print("read image error, ignore the file")

            if img is None:
                cycle_cnt = cycle_cnt - 1
                time.sleep(0.02 * (repeat_time-cycle_cnt))

        if img is not None and img.shape[2] == 1:
            # repeat 3 times to make fake RGB images
            img = np.tile(img, [1, 1, 3])

    return img

def loadImageList(imglist_filepath):
    img_dir = imglist_filepath
    name_list = []
    X = []

    # Loop through the training and test folders, as well as the 'NORMAL' and 'PNEUMONIA' subfolders
    # and append all images into array X.  Append the classification (0 or 1) into array Y.
    #'''
    for fileName in os.listdir(img_dir):
        name_list.append(fileName)
        img = readImageFile(img_dir + fileName)
        if img is not None:
            X.append(img)

        # delete the physical image after read it
        time.sleep(0.01)
        #cmd_str = "rm -f " + img_dir + fileName
        cmd_str = "rm -f " + '"' + img_dir + fileName + '"'
        #print("delete input image file: ", cmd_str)
        os.system(cmd_str)

    return name_list, X


def read_csv_to_list(file):
    label_list = []
    # we only care about the 2 labels -- Image Index / Finding Labels
    if os.path.isfile(file) is True:
        with open(file, "r") as fr:
            reader = csv.reader(fr)
            for line in reader:
                # Image file & label
                label_list.append([line[0], line[1]])
    return label_list

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)


def load_model(model_name):
    if os.path.isfile(model_name):
        print('loading model: %s' % (model_name))

        # info = pickle.dumps(clf, protocol=2)
        # clf = pickle.load(open("new_model.dat", "rb"), encoding="latin1")

        # Note: the previous model was trained in python2 environment
        # We covert it to model which can run with python3 model now
        # clf = pickle.load(open(model_name, "rb"), encoding='latin1')
        clf = pickle.load(open(model_name, "rb"))

        # with open("paper_retina_ref_048_Resnet50_cataract_model_p3.dat", 'wb') as outfile:
        #     pickle.dump(clf, outfile, protocol=-1)
    else:
        raise('No model found...')
    return clf


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='VI CNN')

    parser.add_argument('--input', dest='input_filepath',
                        help='the dataset images folder to be test',
                        default='images', type=str)
    parser.add_argument('--label', dest='label_file',
                        help='the csv file with ground truth for the images, follow the same format as example file',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_filepath',
                        help='the destination folder for output csv file, by default with "outputs" folder',
                        default=None, type=str)
    parser.add_argument('--threshold', dest='threshold_float',
                        help='the threshold for the prediction 1 of the AI model',
                        default=0.5, type=float)
    parser.add_argument(
        "--image_type", type=str, default='diffuse',  # options: fundus, slitlamp, diffuse
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.output_filepath is None:
        output_dir = 'outputs'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        if os.path.isdir(args.output_filepath) is False:
            try:
                os.mkdir(args.output_filepath)
            except:
                print('Wrong output folder name, or not authority to create a new folder !!!')
                exit()
        output_dir = args.output_filepath

    input_images = args.input_filepath
    if input_images is not None:
        if os.path.isfile(input_images):
            test_mode = 'file_mode'
        elif os.path.isdir(input_images):
            print("We only support input as image file currently!!!")
            raise("sample command: python3 main.py --image_type fundus --input ./fundus_images/MS11883R_L.png")
            test_mode = 'path_mode'
        else:
            raise("Wrong input file or folder!!!")
    else:
        raise("Please give the correct input!!!")

    if test_mode == 'path_mode':
        label_model = False
        label_file = args.label_file
        if (label_file is not None) and os.path.isfile(label_file):
            label_model = True
            print('Input has ground truth: {}' .format(label_model))
            assert os.path.splitext(label_file)[-1] == ".csv"

    image_type = args.image_type
    if image_type == 'diffuse':
        model_name = 'EfficientNet_b3'
        my_model_path = 'any048_fold1_class_model4EfficientNet_b3_lr0.0001-AUC9438-statedict.pth'
    elif image_type == 'slitlamp':
        model_name = 'ResNet34'
        my_model_path = 'any048_class_model4ResNet34_lr0.0001_2nd-AUC9338-statedict.pth'
    elif image_type == 'fundus':
        model_name = 'ResNet18'
        my_model_path = 'any048_class_model4ResNet18_xgb131-AUC9703.pth'
        start = time.time()
        xgb_net = load_model(my_model_path)
        end_time = time.time()
        print('loading %s model, time %.2f' % (my_model_path, end_time - start))
    else:
        print("Unsupported image_type: {}".format(image_type))
        raise ("error")

    num_class = 2
    batch_size = 1
    threshold = args.threshold_float

    start = time.time()
    if test_mode == 'path_mode':
        # need to call dataset class to organize images
        DATA_DIR = input_images
        if label_model is True:
            TEST_IMAGE_LIST = read_csv_to_list(label_file)
        else:
            TEST_IMAGE_LIST = None
        data = {
            'test':
                Fundus_Dataset(data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, \
                            transform=None, target_transform=None)
        }
        # Dataloader iterators
        dataloaders = {
            'test': DataLoader(data['test'], batch_size=batch_size, shuffle=False)
        }

        # image_list = []
        # file_list = []
        # Y_list = []
        # y_pred = []
        #
        # step_num = 0
        # sum_dataset = len(dataloaders['test'].dataset)
        #
        # model = get_model(model_name, num_class)
        # # import pdb
        # # pdb.set_trace()
        # # Note: support batch=1 only
        # for imgs, Y, file in tqdm(dataloaders['test']):
        #     Y_list.append(int(Y[0]))
        #     file_list.append(file)
        #
        #     if image_type == 'fundus':
        #         feat_test = feat_extract(model, imgs)
        #         batch_pred = xgb_net.predict_proba(feat_test)
        #         y_pred.append(batch_pred)
        #     else:
        #         pass
        #
        # Y_list = np.array(Y_list, dtype=int)
        # file_list = np.array(file_list)
        # y_pred = np.array(y_pred)
        #
        # y_pred = np.squeeze(y_pred, axis=1)
        # statistic_file = os.path.join(output_dir, 'TestResult.csv')
        # C = open(statistic_file, 'w')
        # C.write(
        #     'file name,AI_model,ground truth,threshold,probability of 0,probability of 1,prediction(0=Non-Cataract 1=Cataract)\n')
        #
        # for i in range(sum_dataset):
        #     probability_0 = y_pred[i, 0]
        #     probability_1 = y_pred[i, 1]
        #     print("filename:{}, probability_0:{}, probability_1:{}".format(file_list[i], probability_0, probability_1))
        #
        #     Threshold_Cataract = threshold
        #     class_result = 0
        #     if probability_1 > Threshold_Cataract:
        #         class_result = 1
        #
        #     C.write('{},{},{},{},{},{},{}\n' \
        #         .format(file_list[i], models[model_index], Y_list[i],Threshold_Cataract,probability_0,probability_1,class_result))
        # C.close()

    elif test_mode == 'file_mode':
        # we only read the input file and give result
        filename = input_images
        model = get_model(model_name, num_class)
        print('\nTest loading model statedict:', model_name)

        if image_type == 'fundus':
            # get model feature
            X_test = []
            X = cv2.resize(cv2.imread(filename), (224, 224))
            X_test.append(X)
            X_test = np.array(X_test, dtype=np.float32)
            feat_test = feat_extract(model, X_test)
            y_pred = xgb_net.predict_proba(feat_test)
            probability_1 = y_pred[0, 1]
        else:
            predict_test = []
            imgs = read_diffuse_slitlamp_File(filename)
            imgs = imgs.unsqueeze(0)

            try:
                model = get_model(model_name, num_class)
                print('\nTest loading model statedict:', my_model_path)
                model = model.to(device=torch.device('cpu'))
                model.load_state_dict(torch.load(my_model_path, map_location=torch.device('cpu')))
                model.eval()
                print('{} statedict model successfully loaded to CPU'.format(my_model_path))


            except Exception as e:
                print('!!! ' + '#==#' * 20 + ' !!!')
                print('Unable to load model to CPU:', e)
                print('!!! ' + '#==#' * 20 + ' !!!')

            features, out1 = model(imgs)

            predict_test.append(out1.detach().cpu().numpy()[:, -1])
            y_pred = predict_test[0]
            probability_1 = y_pred[0]

        probability_0 = 1 - probability_1
        print("filename:{}, probability_0:{}, probability_1:{}" .format(filename, probability_0, probability_1))

        statistic_file = os.path.join(output_dir, 'TestResult.csv')
        if os.path.isfile(statistic_file):
            C = open(statistic_file, 'a+')
        else:
            C = open(statistic_file, 'w')
            C.write('file name,AI_model,threshold,probability of 0,probability of 1,prediction(0=Non-Cataract 1=Cataract)\n')

        Threshold_Cataract = threshold
        class_result = 0
        if probability_1 > Threshold_Cataract:
            class_result = 1

        C.write('{},{},{},{},{},{}\n' \
                .format(filename, models[model_index], Threshold_Cataract, probability_0, probability_1, class_result))
        C.close()

    print("Cataract Validation is Over, please get your results in {} !!!\n" .format(statistic_file))
