from __future__ import print_function
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, ResNet50V2, ResNet101V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc,\
    roc_auc_score, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import pandas as pd
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# tf.debugging.set_log_device_placement(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_img_array(img, size):
    # `img` is a PIL image of size 299x299
    # img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


def load_npy():
    print('-' * 30)
    print("Start .npy loading")
    print('-' * 30)
    # Train_imgs = np.load('npys/Total_Train_image_set_0518.npy')
    # Train_labels = np.load('npys/Total_Train_lable_set_0518.npy')
    # Test_imgs = np.load('npys/Total_Test_image_set_pred.npy')
    # Test_labels = np.load('npys/Total_Test_lable_set_pred.npy')
    Train_imgs = np.load('npys/Total_Train_image_set_331_v4_2_NonM.npy')
    Train_labels = np.load('npys/Total_Train_lable_set_331_v4_2_NonM.npy')
    Test_imgs = np.load('npys/Total_Test_image_set_331_v4_2_NonM_pred.npy')
    Test_labels = np.load('npys/Total_Test_lable_set_331_v4_2_NonM_pred.npy')
    print('-' * 30)
    print("Done")
    print('-' * 30)
    return Train_imgs, Train_labels, Test_imgs, Test_labels


def save_and_display_gradcam(img, heatmap, Number, alpha=0.4):
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    # superimposed_img = superimposed_img[:, :, ::-1]
    # Save the superimposed image
    cam_path = "Gradcam_result/Last_Version_pred/"+str(Number)+".png"
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
    # cv2.imshow("1111", superimposed_img)
    # cv2.waitKey(0)


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
                            input_shape = (train.shape[1], train.shape[2], 3),
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


def Predict(test, test_classes):
    API_model = xcept()
    # sgd = optimizers.SGD(lr=1e-05)
    adam = optimizers.Adam(lr=1e-05)
    # adagrad = optimizers.Adagrad(lr=1e-05)
    API_model.summary()
    API_model.compile(loss="categorical_crossentropy", optimizer=adam,
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    API_model.load_weights('hdf5s/xcept_0917_adam_Transfer_NonM.hdf5')
    print('-' * 30)
    print('Predicting on test data...')
    print('-' * 30)
    imgs_predict_test = API_model.predict(test, verbose=1)
    print("Result_shape: ", imgs_predict_test.shape)
    df = pd.DataFrame(imgs_predict_test)
    df.to_csv('Pred_xcept_0917_adam_Transfer_NonM.csv', index=False)
    np.save('npys/xcept_0917_adam_Transfer_NonM.npy', imgs_predict_test)

    test_classes = np.argmax(test_classes, axis=1)
    imgs_predict_test = np.argmax(imgs_predict_test, axis=1)
    print(test_classes)
    print(imgs_predict_test)

    ############### 4 Class #############

    # result = multilabel_confusion_matrix(test_classes, imgs_predict_test)
    # target_names = ['Benign', 'Nasal_polyp', 'Normal', 'Malignant']
    # cm = classification_report(test_classes, imgs_predict_test, target_names=target_names, digits=3)
    # print(cm)
    # print(result)
    # from sklearn.metrics import precision_recall_fscore_support
    # res = []
    # for l in [0, 1, 2, 3]:
    #     prec, recall, _, _ = precision_recall_fscore_support(np.array(test_classes) == l,
    #                                                          np.array(imgs_predict_test) == l,
    #                                                          pos_label=True, average=None)
    #     res.append([l, recall[0], recall[1]])
    # a = pd.DataFrame(res, columns=['class', 'sensitivity', 'specificity'])
    # print(a)
    # disp = ConfusionMatrixDisplay.from_predictions(test_classes, imgs_predict_test, display_labels=target_names,
    #                                                cmap=plt.cm.Blues, normalize='true', values_format='')
    #
    # n_classes = 4
    # ap = dict()
    # map = 0
    # for i in range(n_classes):
    #     ap[i] = average_precision_score(test_classes_2[:, i], imgs_predict_test_2[:, i])
    #     map += ap[i]
    # print("Benign AP: ", ap[0], "NP AP: ", ap[1], "Normal AP: ", ap[2], "Malignant AP: ", ap[3])
    # print("mAP: ", map/n_classes)

    ############ Non Malignant vs Malignant ##############

    result = multilabel_confusion_matrix(test_classes, imgs_predict_test)
    target_names = ['Non-Malgnant', 'Malignant']
    cm = classification_report(test_classes, imgs_predict_test, target_names=target_names, digits=3)
    print(cm)
    print(result)
    from sklearn.metrics import precision_recall_fscore_support
    res = []
    for l in [0, 1]:
        prec, recall, _, _ = precision_recall_fscore_support(np.array(test_classes) == l,
                                                             np.array(imgs_predict_test) == l,
                                                             pos_label=True, average=None)
        res.append([l, recall[0], recall[1]])
    a = pd.DataFrame(res, columns=['class', 'specificity', 'sensitivity'])
    print(a)
    disp = ConfusionMatrixDisplay.from_predictions(test_classes, imgs_predict_test, display_labels=target_names,
                                                   cmap=plt.cm.Blues, normalize='true', values_format='')

    n_classes = 2
    ap = dict()
    map = 0
    for i in range(n_classes):
        ap[i] = average_precision_score(test_classes_2[:, i], imgs_predict_test_2[:, i])
        map += ap[i]
    print("Non-Malignant AP: ", ap[0], "Malignant AP: ", ap[1])
    print("mAP: ", map / n_classes)


    # ______________AUC______________________
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(test_classes_2.shape)
    print(imgs_predict_test_2.shape)
    test_classes_1 = test_classes.reshape(-1, 1)
    imgs_predict_test_1 = imgs_predict_test.reshape(-1, 1)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_classes_2[:, i], imgs_predict_test_2[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_classes_2.ravel(), imgs_predict_test_2.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label="micro-average ROC curve (area = {0:0.3f})".format(roc_auc["micro"]),
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )
    
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.3f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    lw = 2
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
    for i, color in zip(range(n_classes), colors):
        plt.figure()
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of {0} (area = {1:0.3f})".format(target_names[i], roc_auc[i])
        )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
    
    
    # plt.title("Some extension of Receiver operating characteristic to multiclass")
    # plt.legend(loc="lower right")
    plt.show()
    
    _____________Grad Cam_________________
    Prepare image
    img_size = (train.shape[1], train.shape[2])
    preprocess_input = tf.keras.applications.vgg19.preprocess_input
    decode_predictions = tf.keras.applications.vgg19.decode_predictions
    
    # Remove last layer's softmax
    API_model.layers[-1].activation = None

    
    img_d = test[0]
    img_d = img_d[:, :, ::-1]
    img_array = preprocess_input(get_img_array(img_d, size=img_size))
    # heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv4')
    # plt.matshow(heatmap)
    disp.plot()
    plt.show()
    # save_and_display_gradcam(img_d, heatmap)
    
    for p in range(test.shape[0]):
        img_d = test[p]
        img_d = img_d[:, :, ::-1]
        img_array = preprocess_input(get_img_array(img_d, size=img_size))
        heatmap = make_gradcam_heatmap(img_array, API_model, 'block14_sepconv2_act')
        plt.matshow(heatmap)
        # disp.plot()
        # plt.show()
        save_and_display_gradcam(img_d, heatmap, p)
    


def normalization(X):
    X_len = X[0]
    X_array = np.ndarray(X.shape)
    for i in X_len:
        Y = cv2.normalize(i, None, 0, 256, cv2.NORM_MINMAX)
        X_array[i] = Y
    return X_array


if __name__ == '__main__':

    with tf.device("/gpu:1"):
        
        img_rows = 331
        img_cols = 331
        classes = ["NP", "Malignant"]

        train, train_classes, test_img, test_classes = load_npy()
        train_X, val_X, train_Y, val_Y = train_test_split(train, train_classes, test_size=0.4, random_state=123)

        train_Class = tf.keras.utils.to_categorical(train_Y)
        test_Class = tf.keras.utils.to_categorical(test_classes)
        Val_Class = tf.keras.utils.to_categorical(val_Y)

        Predict(test_img, test_Class)

