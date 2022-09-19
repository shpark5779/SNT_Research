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
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay

tf.debugging.set_log_device_placement(True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def Generator_Dataset(PATH, Test_PATH):
    print('-' * 30)
    print('Creating Train Data Set...')
    print('-' * 30)
    train_total_lable_list = np.ndarray(1200 * 4, dtype=np.uint8)
    train_total_image = np.ndarray((1200 * 4, img_cols, img_rows, 3), dtype=np.uint8)
    class_list = os.listdir(PATH)
    print(class_list)
    j = 0
    for class_name in class_list:
        train_image_list = os.listdir(PATH + class_name + "/")
        print(len(train_image_list))
        train_image_list = train_image_list[:1200]
        print(len(train_image_list))
        for train_image_name in train_image_list:
            train_img = cv2.imread(os.path.join(PATH + class_name, train_image_name), cv2.IMREAD_COLOR)
            train_total_image[j] = train_img
            train_total_lable_list[j] = class_list.index(class_name)
            if j % 100 == 0:
                print(class_name + ' Done: {0}/{1} images'.format(j, len(train_image_list)))
            j += 1
    print("Classes: ", np.unique(train_total_lable_list))


    B_list = os.listdir(Test_PATH + "Benign/")
    M_list = os.listdir(Test_PATH + "Malignant/")
    N_list = os.listdir(Test_PATH + "Normal/")
    NP_list = os.listdir(Test_PATH + "NP/")
    list_len = len(B_list)+len(M_list)+len(N_list)+len(NP_list)
    print("list_len= ", list_len)
    print('-' * 30)
    print('Creating Test Data Set...')
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
            total_image[j] = img
            total_label_list[j] = class_list.index(class_name)
            if j % 100 == 0:
                print(class_name + ' Done: {0}/{1} images'.format(j, list_len))
                print(total_label_list[j])
            j += 1

    print("Total_train_volume= ", j)
    # total_lable_list = np.delete(total_lable_list, 0)
    # total_image = np.delete(total_image, 0)
    print("Train_image_set_shape: ", train_total_image.shape, "Train_lable_set_shape: ", train_total_lable_list.shape)
    np.save('npys/Total_Train_image_set_v2.npy', train_total_image)
    # np.save('Cropped_Image_Set.npy', total_image[:, 141:-141, 141:-141, :])
    np.save('npys/Total_Train_lable_set_v2.npy', train_total_lable_list)
    # np.save('Original_to_resize_images_v2.npy', total_resize)

    print('Saving to Train .npy files done.')

    print("list_len= ", list_len)
    print("Total_volume= ", j)
    # total_lable_list = np.delete(total_lable_list, 0)
    # total_image = np.delete(total_image, 0)
    print("Test_image_set_shape: ", total_image.shape, "Test_lable_set_shape: ", total_label_list.shape)
    np.save('npys/Total_Test_image_set.npy', total_image)
    # np.save('Cropped_Image_Set.npy', total_image[:, 141:-141, 141:-141, :])
    np.save('npys/Total_Test_lable_set.npy', total_label_list)
    # np.save('Original_to_resize_images_v2.npy', total_resize)
    print("Classes: ", np.unique(total_label_list))
    print(total_label_list)
    print('Saving to .npy files done.')


def Generator_Test_Dataset(PATH, class_list):
    print('-' * 30)
    print('Creating Data Set...')
    print('-' * 30)
    total_lable_list = np.ndarray(6400 * 2, dtype=np.uint8)
    total_image = np.ndarray((6400 * 2, img_cols, img_rows, 3), dtype=np.uint8)

    print(class_list)
    j = 0
    for class_name in class_list:
        image_list = os.listdir(PATH + class_name + "/")
        print(len(image_list))
        image_list = image_list[:6400]
        print(len(image_list))
        for image_name in image_list:
            img = cv2.imread(os.path.join(PATH + class_name, image_name), cv2.IMREAD_COLOR)
            total_image[j] = img
            total_lable_list[j] = class_list.index(class_name)
            if j % 99 == 0:
                print(class_name + ' Done: {0}/{1} images'.format(j, len(image_list)))
            j += 1

    print("Total_volume= ", j)
    # total_lable_list = np.delete(total_lable_list, 0)
    # total_image = np.delete(total_image, 0)
    print("image_set_shape: ", total_image.shape, "lable_set_shape: ", total_lable_list.shape)
    np.save('npys/NPM_Testing_image_set.npy', total_image)
    # np.save('Cropped_Image_Set.npy', total_image[:, 141:-141, 141:-141, :])
    np.save('npys/NPM_Testing_lable_set.npy', total_lable_list)
    # np.save('Original_to_resize_images_v2.npy', total_resize)
    print("Classes: ", np.unique(total_lable_list))
    print('Saving to .npy files done.')


def load_testing_npy():
    print('-' * 30)
    print("Start .npy loading")
    print('-' * 30)
    # imgs = np.load("Cropped_Image_Set.npy")
    # imgs = np.load("Total_image_set.npy")
    imgs = np.load('npys/NPM_Testing_image_set.npy')
    lables = np.load('npys/NPM_Testing_lable_set.npy')
    print('-' * 30)
    print("Done")
    print('-' * 30)
    return imgs, lables


def load_npy():
    print('-' * 30)
    print("Start .npy loading")
    print('-' * 30)
    # imgs = np.load("Cropped_Image_Set.npy")
    # imgs = np.load("Total_image_set.npy")
    Train_imgs = np.load("npys/Total_Train_image_set_v2.npy")
    Train_labels = np.load("npys/Total_Train_lable_set_v2.npy")
    Test_imgs = np.load("npys/Total_Test_image_set.npy")
    Test_labels = np.load("npys/Total_Test_lable_set.npy")
    print('-' * 30)
    print("Done")
    print('-' * 30)
    return Train_imgs, Train_labels, Test_imgs, Test_labels


def kfold_learning(train, classes):
    with tf.device('/GPU:1'):
        imgs, lables = load_npy()
        # train_X, train_Y, test_X, test_Y = train_test_split(imgs, lables, test_size=0.2, random_state=123)
        # train_X, train_Y, val_X, val_Y = train_test_split(train_X, train_Y, test_size=0.4, random_state=321)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        k = 0
        for train_ix, test_ix in kfold.split(train, classes):
            train_X, test_X = imgs[train_ix], imgs[test_ix]
            train_Y, test_Y = lables[train_ix], lables[test_ix]
            train_y = tf.keras.utils.to_categorical(train_y, num_classes=8)
            test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=8)

            API_model = ResNet152V2(include_top=False, weights=None, input_tensor=None,
                                    input_shape=(img_cols, img_rows, 3),
                                    pooling=None, classes=8)
            flat1 = Flatten()(API_model.layers[-1].output)
            class1 = Dense(1024, activation='relu')(flat1)
            output = Dense(8, activation='softmax')(class1)
            model = Model(inputs=API_model.inputs, outputs=output)
            sgd = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
            model.summary()
            model.compile(loss="categorical_crossentropy", optimizer=sgd)
            model_checkpoint = ModelCheckpoint('test_' + str(k) + '.hdf5', monitor='loss', save_best_only=True)
            model.fit(train_X, train_y, batch_size=1, epochs=1000, callbacks=model_checkpoint)
            k = k + 1


def normal_learning(train, classes, val, val_C):
    API_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
                      input_shape=(train.shape[1], train.shape[2], 3),
                      pooling=None, classes=4)
    flat1 = Flatten()(API_model.layers[-1].output)
    class1 = Dense(4096, activation='relu')(flat1)
    class2 = Dense(4096, activation='relu')(class1)
    output = Dense(4, activation='softmax')(class2)
    model = Model(inputs=API_model.inputs, outputs=output)
    sgd = optimizers.SGD(lr=1e-05)
    adam = optimizers.Adam(lr=1e-05)
    adagrad = optimizers.Adagrad(lr=1e-05)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    tensorboard = TensorBoard(log_dir="logs_test/InceptionResNet_02_09_sgd_transfer_v3/", histogram_freq=1, write_graph=True)
    model_checkpoint = ModelCheckpoint('HDF5s/InceptionResNet_02_09_sgd_transfer_v3.hdf5', monitor='loss', save_best_only=True)
    earlystopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=0,
                                  mode="min", restore_best_weights=True)
    model.fit(train, classes, validation_data=(val, val_C), batch_size=192, epochs=5000, shuffle=True,
              callbacks=[tensorboard, model_checkpoint, earlystopping])


def Evaluate(test_X, test_Class):
    API_model = ResNet152V2(include_top=False, weights=None, input_tensor=None,
                            input_shape=(train_img.shape[1], train_img.shape[2], 3),
                            pooling=None, classes=8)
    flat1 = Flatten()(API_model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(8, activation='softmax')(class1)
    model = Model(inputs=API_model.inputs, outputs=output)
    sgd = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    train_weight = model.load_weights('test_.hdf5')
    test_loss, test_acc = model.evaluate(x=test_X, y=test_Class, sample_weight=train_weight)
    print('\n테스트 Loss:', test_loss, '\n테스트 정확도:', test_acc)


def Predict(test, test_classes):
    API_model = VGG19(include_top=False, weights='imagenet', input_tensor=None,
                      input_shape=(train_img.shape[1], train_img.shape[2], 3),
                      pooling=None, classes=4)
    flat1 = Flatten()(API_model.layers[-1].output)
    class1 = Dense(4096, activation='relu')(flat1)
    class2 = Dense(4096, activation='relu')(class1)
    output = Dense(4, activation='softmax')(class2)
    model = Model(inputs=API_model.inputs, outputs=output)
    # sgd = optimizers.SGD(lr=1e-05)
    adam = optimizers.Adam(lr=1e-05)
    # adagrad = optimizers.Adagrad(lr=1e-05)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=adam,
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('VGG19_01_18_adam_transfer_v3.hdf5')
    print('-' * 30)
    print('Predicting on test data...')
    print('-' * 30)
    imgs_predict_test = model.predict(test, verbose=1)
    print("Result_shape: ", imgs_predict_test.shape)
    np.save('npys/Pridict_Result.npy', imgs_predict_test)
    # result = tf.math.confusion_matrix(
    #     test_classes, imgs_predict_test, num_classes=4, weights=None, dtype=tf.dtypes.int32, name=None)
    # result = confusion_matrix(test_classes, imgs_predict_test)
    print(test_classes)
    print(imgs_predict_test)
    test_classes = np.argmax(test_classes, axis=1)
    imgs_predict_test = np.argmax(imgs_predict_test, axis=1)
    print(test_classes)
    print(imgs_predict_test)
    result = multilabel_confusion_matrix(test_classes, imgs_predict_test)
    target_names = ['Benign', 'Malignant', 'Normal', 'Nasal_polyp']
    cm = classification_report(test_classes, imgs_predict_test, target_names=target_names)
    print(cm)
    print(result[0])
    disp = ConfusionMatrixDisplay.from_estimator(model, test_classes, imgs_predict_test, display_labels=target_names,
                                                 cmap=plt.cm.Blues, normalize='Normalized confusion matrix')
    disp.plot()
    plt.show()


def normalization(X):
    X_len = X[0]
    X_array = np.ndarray(X.shape)
    for i in X_len:
        Y = cv2.normalize(i, None, 0, 256, cv2.NORM_MINMAX)
        X_array[i] = Y
    return X_array


if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError as e:
    #         # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    #         print(e)

    # class별 1000장씩 8000장
    train_data_path = "data/processed_data_v2/"
    test_data_path = "data/Test_img/"
    # Folder_name = ["dyed-lifted-polyps/", "dyed-resection-margins/", "esophagitis/", "normal-cecum/", "normal-pylorus/",
    #                "normal-z-line/", "polyps/", "ulcerative-colitis/"]
    img_rows = 224
    img_cols = 224
    classes = ["NP", "Malignant"]
    # Generator_Test_Dataset(data_path, classes)
    # Generator_Dataset(train_data_path, test_data_path)
    # train, classes = load_testing_npy()
    # train = normalization(train)

    train_img, train_classes, test_img, test_classes = load_npy()
    print(train_img.shape, test_img.shape)
    train_X, val_X, train_Y, val_Y = train_test_split(train_img, train_classes, test_size=0.4, random_state=123)
    # train_img, test_img, train_Lable, test_Lable = train_test_split(train_X, train_Y, test_size=0.14, random_state=222)
    # print("Train_img_set_Shape: ", train_img.shape)
    # print("Validation_img_set_Shape: ", val_img.shape)
    # print("Test_img_set_Shape: ", test_X.shape)
    # print(train_Lable)
    # print(np.unique(train_Lable, return_counts=True))
    # print(np.unique(val_Lable, return_counts=True))
    # print(np.unique(test_Y, return_counts=True))
    # train_Class = tf.keras.utils.to_categorical(train_Lable)
    # test_Class = tf.keras.utils.to_categorical(test_Y)
    # val_Class = tf.keras.utils.to_categorical(val_Lable)

    print("Train_img_set_Shape: ", train_img.shape)
    print("Validation_img_set_Shape: ", val_X.shape)
    print("Test_img_set_Shape: ", test_img.shape)
    print(train_Y)
    print(np.unique(train_Y, return_counts=True))
    print(np.unique(test_classes, return_counts=True))
    train_Class = tf.keras.utils.to_categorical(train_Y)
    test_Class = tf.keras.utils.to_categorical(test_classes)
    Val_Class = tf.keras.utils.to_categorical(val_Y)

    # kfold_learning(train_X, train_Class) # Generator_Dataset 끄고 실행할 것
    normal_learning(train_X, train_Class, val_X, Val_Class)
    # Predict(test_img, test_Class)
    # Evaluate(test_X, test_Class)
