from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_npy():
    print('-' * 30)
    print("Start .npy loading")
    print('-' * 30)
    Train_imgs = np.load('npys/Total_Train_image_set_331_v4_2_NonM.npy')
    Train_labels = np.load('npys/Total_Train_lable_set_331_v4_2_NonM.npy')
    Test_imgs = np.load('npys/Total_Test_image_set_331_v4_2_NonM_pred.npy')
    Test_labels = np.load('npys/Total_Test_lable_set_331_v4_2_NonM_pred.npy')
    print('-' * 30)
    print("Done")
    print('-' * 30)
    return Train_imgs, Train_labels, Test_imgs, Test_labels


def inceptionres():
    API_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                  input_shape=(train.shape[1], train.shape[2], 3),
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
                      input_shape=(train.shape[1], train.shape[2], 3),
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
                            input_shape=(train.shape[1], train.shape[2], 3),
                            pooling=None, classes=4)
    x = API_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    model.summary()
    return model


def xcept():
    API_model = Xception(include_top=False, weights='imagenet', input_tensor=None,
                         input_shape=(train.shape[1], train.shape[2], 3),
                         pooling=None, classes=2)
    x = API_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    model.summary()
    return model


def res152():
    API_model = ResNet152V2(include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=(train.shape[1], train.shape[2], 3),
                            pooling=None, classes=4)
    x = API_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    model.summary()
    return model


def Predict(test, weight_path):
    API_model = xcept()
    adam = optimizers.Adam(lr=1e-05)
    API_model.compile(loss="categorical_crossentropy", optimizer=adam,
                      metrics=[metrics.mae, metrics.categorical_accuracy])
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    API_model.load_weights(weight_path)
    print('-' * 30)
    print('Predicting on test data...')
    print('-' * 30)
    imgs_predict_test = API_model.predict(test, verbose=1)
    print('-' * 30)
    print('Done')
    print('-' * 30)
    return imgs_predict_test

def text_to_array(res):
    row = -1
    ap = np.ndarray((4, 4))
    for t in res.text_:
        row += 1
        n = -1
        for pp in t:
            n += 1
            ap[row, n] = float(pp.get_text())
            print('row = ', row, 'col = ', n)
    return ap

def text_to_array_NonM(res):
    row = -1
    ap = np.ndarray((2, 2))
    for t in res.text_:
        row += 1
        n = -1
        for pp in t:
            n += 1
            ap[row, n] = float(pp.get_text())
            print('row = ', row, 'col = ', n)
    return ap





if __name__ == '__main__':
    with tf.device("/gpu:1"):
        data_path = "data/processed_data_v4/"
        test_data_path = "data/Processed_test_imgs/"

        img_rows = 331
        img_cols = 331
        classes = ["NP", "Malignant"]

        train, train_classes, test_img, test_classes = load_npy()
        img_s = train[0]
        img_s = img_s[:, :, ::-1]

        print(train.shape)
        train_X, val_X, train_Y, val_Y = train_test_split(train, train_classes, test_size=0.4, random_state=123)

        print("Train_img_set_Shape: ", train_X.shape)
        print("Validation_img_set_Shape: ", val_X.shape)
        print("Test_img_set_Shape: ", test_img.shape)
        print(train_Y)
        print(np.unique(train_Y, return_counts=True))
        print(np.unique(test_classes, return_counts=True))
        train_Class = tf.keras.utils.to_categorical(train_Y)
        test_Class = tf.keras.utils.to_categorical(test_classes)
        Val_Class = tf.keras.utils.to_categorical(val_Y)

        model1 = Predict(test_img, 'hdf5s/xcept_0917_adam_Transfer_NonM.hdf5')
        model2 = Predict(test_img, 'hdf5s/xcept_0917_adam_Transfer_v4_1_MonM.hdf5')
        model3 = Predict(test_img, 'hdf5s/xcept_0917_adam_Transfer_v4_2_NonM.hdf5')
        model4 = Predict(test_img, 'hdf5s/xcept_0914_adam_Transfer_v4_3_NonM.hdf5')
        model5 = Predict(test_img, 'hdf5s/xcept_0917_adam_Transfer_v4_5_MonM.hdf5')
        model6 = Predict(test_img, 'hdf5s/xcept_0917_adam_Transfer_v4_6_NonM.hdf5')

        test_classes = np.argmax(test_Class, axis=1)
        model1 = np.argmax(model1, axis=1)
        model2 = np.argmax(model2, axis=1)
        model3 = np.argmax(model3, axis=1)
        model4 = np.argmax(model4, axis=1)
        model5 = np.argmax(model5, axis=1)
        model6 = np.argmax(model6, axis=1)

        # target_names = ['Benign', 'Nasal_polyp', 'Normal', 'Malignant']
        target_names = ['Non-Malignant', 'Malignant']

        conf1 = ConfusionMatrixDisplay.from_predictions(test_classes, model1, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')
        conf2 = ConfusionMatrixDisplay.from_predictions(test_classes, model2, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')
        conf3 = ConfusionMatrixDisplay.from_predictions(test_classes, model3, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')
        conf4 = ConfusionMatrixDisplay.from_predictions(test_classes, model4, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')
        conf5 = ConfusionMatrixDisplay.from_predictions(test_classes, model5, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')
        conf6 = ConfusionMatrixDisplay.from_predictions(test_classes, model6, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')
        plt.show()

        # conf1 = text_to_array(conf1)
        # conf2 = text_to_array(conf2)
        # conf3 = text_to_array(conf3)
        # conf4 = text_to_array(conf4)
        # conf5 = text_to_array(conf5)
        # conf6 = text_to_array(conf6)

        conf1 = text_to_array_NonM(conf1)
        conf2 = text_to_array_NonM(conf2)
        conf3 = text_to_array_NonM(conf3)
        conf4 = text_to_array_NonM(conf4)
        conf5 = text_to_array_NonM(conf5)
        conf6 = text_to_array_NonM(conf6)

        conf_mean = np.mean([conf1, conf2, conf3, conf4, conf5, conf6], axis=0)
        conf_std = np.std([conf1, conf2, conf3, conf4, conf5, conf6], axis=0)

        print(conf_mean)
        df_conf_mean = pd.DataFrame(conf_mean, columns=target_names)
        print(df_conf_mean)
        res = sn.heatmap(df_conf_mean, annot=True, vmin=0.0, vmax=1.0,
                         fmt='.3f', cmap="Blues", cbar_kws={"shrink": .82},
                         linewidths=0.1, linecolor='gray', yticklabels=target_names)
        res.set_ylabel('True label')
        res.set_xlabel('Predicted label')
        # res.invert_yaxis()
        print(conf_std)
        n = -1
        col = -1
        # row = 0
        for t in res.texts:
            print(t)
            n += 1
            col += 1
            row = n // 2
            if n % 2 == 0:
                if row == 2:
                    pass
                else:
                    col = 0
                    print(n)
                    print('row = ', row, 'col = ', col)
                    t.set_text(t.get_text() + "±" + str(format(conf_std[row, col], ".2f")))
                    print(t.get_text() + "±" + str(format(conf_std[row, col], ".2f")))
            else:
                if row == 2:
                    pass
                else:
                    print(n)
                    print('row = ', row, 'col = ', col)
                    t.set_text(t.get_text() + "±" + str(format(conf_std[row, col], ".2f")))

        plt.show()
