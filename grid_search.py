# from __future__ import print_function
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, ResNet50V2, ResNet101V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def Generate_Train_Dataset(Test_PATH):
    print('-' * 30)
    print('Creating Train Data Set...')
    print('-' * 30)
    class_list = os.listdir(Test_PATH)
    print(class_list)

    B_list = os.listdir(Test_PATH + "Benign/")
    M_list = os.listdir(Test_PATH + "Malignant/")
    N_list = os.listdir(Test_PATH + "Normal/")
    NP_list = os.listdir(Test_PATH + "NP/")
    list_len = len(B_list) + len(M_list) + len(N_list) + len(NP_list)
    print("list_len= ", list_len)
    print('-' * 30)
    print('Creating Train Data Set...')
    print('-' * 30)
    # class_list = os.listdir(Test_PATH)
    print(class_list)
    total_label_list = np.ndarray(list_len, dtype=np.uint8)
    total_image = np.ndarray((list_len, img_cols, img_rows, 3), dtype=np.uint8)
    j = 0
    for class_name in class_list:
        image_list = os.listdir(Test_PATH + class_name + "/")
        print(len(image_list))

        image_list = image_list[:len(image_list)]
        print(len(image_list))
        for image_name in image_list:
            img = cv2.imread(os.path.join(Test_PATH + class_name, image_name), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_cols, img_rows))
            total_image[j] = img
            total_label_list[j] = class_list.index(class_name)
            if j % 100 == 0:
                print(class_name + ' Done: {0}/{1} images'.format(j, list_len))
                print(total_label_list[j])
            j += 1

    print("list_len= ", list_len)
    print("Total_volume= ", j)
    print("Train_image_set_shape: ", total_image.shape, "Train_lable_set_shape: ", total_label_list.shape)
    np.save('npys/Total_Train_image_set_0518_224.npys', total_image)
    np.save('npys/Total_Train_lable_set_0518_224.npys', total_label_list)
    print("Train_Classes: ", np.unique(total_label_list))
    print(total_label_list)
    print('Saving to .npys files done.')


def Generate_Test_Dataset(Test_PATH):
    class_list = os.listdir(Test_PATH)
    B_list = os.listdir(Test_PATH + "Benign/")
    M_list = os.listdir(Test_PATH + "Malignant/")
    N_list = os.listdir(Test_PATH + "Normal/")
    NP_list = os.listdir(Test_PATH + "NP/")
    list_len = len(B_list) + len(M_list) + len(N_list) + len(NP_list)
    print("list_len= ", list_len)
    print('-' * 30)
    print('Creating Test Data Set...')
    print('-' * 30)
    print(class_list)
    total_label_list = np.ndarray(list_len, dtype=np.uint8)
    total_image = np.ndarray((list_len, img_cols, img_rows, 3), dtype=np.uint8)
    j = 0
    for class_name in class_list:
        image_list = os.listdir(Test_PATH + class_name + "/")
        print(len(image_list))

        image_list = image_list[:len(image_list)]
        print(len(image_list))
        for image_name in image_list:
            img = cv2.imread(os.path.join(Test_PATH + class_name, image_name), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_cols, img_rows))
            total_image[j] = img
            total_label_list[j] = class_list.index(class_name)
            if j % 100 == 0:
                print(class_name + ' Done: {0}/{1} images'.format(j, list_len))
                print(total_label_list[j])
            j += 1

    print("list_len= ", list_len)
    print("Total_volume= ", j)
    print("Test_image_set_shape: ", total_image.shape, "Test_lable_set_shape: ", total_label_list.shape)
    np.save('npys/Total_Test_image_set_224_pred.npys', total_image)
    np.save('npys/Total_Test_lable_set_224_pred.npys', total_label_list)
    print("Test_Classes: ", np.unique(total_label_list))
    print(total_label_list)
    print('Saving to .npys files done.')


def load_testing_npy():
    print('-' * 30)
    print("Start .npys loading")
    print('-' * 30)
    Train_imgs = np.load("npys/Total_Train_image_set_331_v4_2_NonM.npy")
    Train_labels = np.load("npys/Total_Train_lable_set_331_v4_2_NonM.npy")
    Test_imgs = np.load("npys/Total_Test_image_set_331_v4_2_NonM_pred.npy")
    Test_labels = np.load("npys/Total_Test_lable_set_331_v4_2_NonM_pred.npy")
    print('-' * 30)
    print("Done")
    print('-' * 30)
    return Train_imgs, Train_labels, Test_imgs, Test_labels


def load_npy():
    print('-' * 30)
    print("Start .npys loading")
    print('-' * 30)
    Train_imgs = np.load("npys/Total_Train_image_set_0518.npy")
    Train_labels = np.load("npys/Total_Train_lable_set_0518.npy")
    Test_imgs = np.load("npys/Total_Test_image_set_pred.npy")
    Test_labels = np.load("npys/Total_Test_lable_set_pred.npy")
    print('-' * 30)
    print("Done")
    print('-' * 30)
    return Train_imgs, Train_labels, Test_imgs, Test_labels


def incepres():
    API_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                  input_shape=(train_img.shape[1], train_img.shape[2], 3),
                                  pooling=None, classes=4)
    pool = GlobalAveragePooling2D()(API_model.layers[-1].output)
    # flat1 = Flatten()(API_model.layers[-1].output)
    # class1 = Dense(4096, activation='relu')(flat1)
    # class2 = Dense(4096, activation='relu')(class1)
    output = Dense(4, activation='softmax')(pool)
    model = Model(inputs=API_model.inputs, outputs=output)
    return model


def vgg_19():
    API_model = VGG19(include_top=False, weights='imagenet', input_tensor=None,
                      input_shape=(train_img.shape[1], train_img.shape[2], 3),
                      pooling=None, classes=4)
    # pool = GlobalAveragePooling2D()(API_model.layers[-1].output)
    flat1 = Flatten()(API_model.layers[-1].output)
    class1 = Dense(4096, activation='relu')(flat1)
    class2 = Dense(4096, activation='relu')(class1)
    output = Dense(4, activation='softmax')(class2)
    model = Model(inputs=API_model.inputs, outputs=output)
    return model


def naslarge():
    API_model = NASNetLarge(include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=(train_img.shape[1], train_img.shape[2], 3),
                            pooling=None, classes=4)
    x = API_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    # model.summary()
    return model


def xcept():
    API_model = Xception(include_top=False, weights='imagenet', input_tensor=None,
                         input_shape=(train_img.shape[1], train_img.shape[2], 3),
                         pooling=None, classes=4)
    x = API_model.output
    # x = BatchNormalization
    x = GlobalAveragePooling2D()(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    # model.summary()
    return model


def res152():
    API_model = ResNet152V2(include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=(train_img.shape[1], train_img.shape[2], 3),
                            pooling=None, classes=4)
    x = API_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    # model.summary()
    return model


def class_weight(label):
    benign = list(label).count(0)
    Malignant = list(label).count(1)
    Normal = list(label).count(2)
    NP = list(label).count(3)

    weight_for_0 = (1 / benign) * (label.shape[0] / 4)
    weight_for_3 = (1 / Malignant) * (label.shape[0] / 4)
    weight_for_2 = (1 / Normal) * (label.shape[0] / 4)
    weight_for_1 = (1 / NP) * (label.shape[0] / 4)
    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3}
    print('Weight for class Benign: {:.2f}'.format(weight_for_0), 'Weight for class NP: {:.2f}'.format(weight_for_1),
          'Weight for class Normal: {:.2f}'.format(weight_for_2),
          'Weight for class Malignant: {:.2f}'.format(weight_for_3))
    return class_weight


def NonM_class_weight(label):
    Non_Malignant = list(label).count(0)
    Malignant = list(label).count(1)

    weight_for_1 = (1 / Malignant) * (label.shape[0] / 2)
    weight_for_0 = (1 / Non_Malignant) * (label.shape[0] / 2)

    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class Non Malignant: {:.2f}'.format(weight_for_0),
          'Weight for class Malignant: {:.2f}'.format(weight_for_1))
    return class_weight

def create_model(learn_rate=0.045):
    API_model = xcept()
    Optimizer = optimizers.Adam(lr=learn_rate)
    # adam = optimizers.Adam(lr=Learning_rate)
    # adagrad = optimizers.Adagrad(lr=Learning_rate)
    API_model.compile(loss="categorical_crossentropy", optimizer=Optimizer,
                      metrics=["accuracy"])
    return API_model  
  

if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError as e:
    #         # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    #         print(e)

    # class별 1000장씩 8000장
    data_path = "data/processed_data_v4/"
    test_data_path = "data/Processed_test_imgs/"

    img_rows = 331
    img_cols = 331
    
    #____________Creat npy for learning_______________
    # Generate_Train_Dataset(data_path)
    # Generate_Test_Dataset(test_data_path)
    
    #_____________Load npy data_____________________
    train_img, train_classes, test_img, test_classes = load_npy()

    #___________________Class weight______________
    # class_weight = NonM_class_weight(train_classes)
    class_weight = class_weight(train_classes)

    # ___________________Train______________
    train_X, val_X, train_Y, val_Y = train_test_split(train_img, train_classes, test_size=0.4, random_state=123)

    train_Class = tf.keras.utils.to_categorical(train_Y)
    test_Class = tf.keras.utils.to_categorical(test_classes)
    Val_Class = tf.keras.utils.to_categorical(val_Y)

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        'F1-score': make_scorer(f1_score)
    }
    model = KerasClassifier(build_fn=create_model, epochs=5, verbose=1)
    batch_size = [10, 15, 20, 30]
    learn_rate = [1e-05, 1e-04, 1e-03, 1e-02]
    param_grid = dict(batch_size=batch_size, learn_rate=learn_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None,
                        cv=3, refit=True, return_train_score=True, verbose=3)
    grid_result = grid.fit(train_X, train_Class, class_weight=class_weight)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    import pandas as pd
    df = pd.DataFrame(grid_result.cv_results_)
    df.to_csv('Xception_Grid_result.csv')
    df1 = pd.DataFrame(grid_result.cv.best_result_)
    df1.to_csv('Xception_Grid_best_result.csv')
    df2 = pd.DataFrame(grid_result.cv.best_estimator_)
    df1.to_csv('Xception_Grid_best_estimator.csv')
    df3 = pd.DataFrame(grid_result.cv.best_score_)
    df1.to_csv('Xception_Grid_best_score.csv')
    df4 = pd.DataFrame(grid_result.cv.scorer_)
    df1.to_csv('Xception_Grid_scorer.csv')
