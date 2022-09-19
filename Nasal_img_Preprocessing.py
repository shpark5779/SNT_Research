import cv2
import os

import matplotlib.pyplot as plt
import numpy as np

Be = 0
Ma = 0
No = 0
nP = 0


def Contour_process(IMAGE):
    IMAGE = IMAGE[5:-5, 5:-5]
    imgcopy = IMAGE.copy()
    img_gray = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img_gray, 35, 255, cv2.THRESH_BINARY)
    # thr = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    contours1,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x, y, w, h = cv2.boundingRect(contours1[-1])
    x_dis = 0
    y_dis = 0
    Cnum = 0
    for k in range(len(contours1)):
        x, y, w, h = cv2.boundingRect(contours1[k])
        print("Contour_num= ", k, "x_dis= ", w, "y_dis= ", h)
        if x_dis < w:
            if y_dis < h:
                x_dis, y_dis = w - x, h - y
                x_1, y_1, w_1, h_1 = x, y, w, h
                Cnum = k

    stencil_r = np.zeros(IMAGE.shape[:-1]).astype(np.uint8)
    stencil_g = np.zeros(IMAGE.shape[:-1]).astype(np.uint8)
    stencil_b = np.zeros(IMAGE.shape[:-1]).astype(np.uint8)
    stencil = cv2.merge([stencil_r, stencil_g, stencil_b])

    cv2.drawContours(IMAGE, contours1[Cnum], -1, (0, 0, 255), 3)
    # ellipse = cv2.fitEllipse(contours1[Cnum])
    # stencil = cv2.ellipse(stencil, ellipse, (255, 255, 255), -1)
    (i, j), r = cv2.minEnclosingCircle(contours1[Cnum])
    stencil = cv2.circle(stencil, (int(i), int(j)), int(r) - 21, (255, 255, 255), -1)
    draw_ellipse = stencil.copy()
    ellipse_gray = cv2.cvtColor(draw_ellipse, cv2.COLOR_BGR2GRAY)
    contours2, _ = cv2.findContours(ellipse_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    x_2, y_2, w_2, h_2 = cv2.boundingRect(contours2[0])
    sel = stencil != 0  # select everything that is not mask_value
    stencil[sel] = imgcopy[sel]  # and fill it with fill_color
    crop = stencil[y_2:y_2 + h_2, x_2:x_2 + w_2]
    return crop


def processing(Hospital_name, CLASS, DETAILS, data_path, LIST):
    global Be
    global Ma
    global No
    global nP
    processed_list = os.listdir("data/by hospital/Processed_img")
    LIST = list(set(LIST).intersection(processed_list))
    if CLASS == "Benign":
        Be = Be + len(LIST)
    elif CLASS == "Malignant":
        Ma = Ma + len(LIST)
    elif CLASS == "Normal":
        No = No + len(LIST)
    elif CLASS == "NP":
        nP = nP + len(LIST)

    image_shape = []
    N = 0
    for image_name in LIST:
        image = cv2.imread(data_path + image_name, cv2.IMREAD_COLOR)
        print("*" * 30)
        print(data_path + image_name)
        print("*" * 30)
        pred_img = Contour_process(image)
        image_name = image_name.split(".")[0]+".png"
        if DETAILS == "None":
            cv2.imwrite("data/processed_data/" + CLASS + "/" + image_name, pred_img)
        else:
            cv2.imwrite("data/processed_data/" + CLASS + "/" + DETAILS + "/" + image_name, pred_img)
        image_shape.append(image.shape)
        N = N + 1
    a = set(image_shape)
    # cv2.imshow("a",image)
    # cv2.waitKey(10)
    print('*' * 30)
    print(Hospital_name + CLASS + ' Done')
    print(a)
    print('*' * 30)
    print(Be, Ma, No, nP)
    print('*' * 30)
    return Be, Ma, No, nP


def read_data_path(PATH):
    Class_list = ['Benign', 'Malignant', 'NP', 'Normal']
    folder_list = os.listdir(PATH)
    folder_list = sorted(folder_list)
    print(folder_list)
    for folder_name in folder_list:
        Classes_folder = os.listdir(PATH + '/' + folder_name + '/')
        print(folder_name)
        print(Classes_folder)
        for Class in Class_list:
            if Class in Classes_folder:
                if Class == 'Benign':
                    Benign_list = ['IP', 'Non-vascular, etc', 'Vascular']
                    for Benign_Class in Benign_list:
                        Benign_path = PATH + '/' + folder_name + '/' + Class + '/' + Benign_Class + '/'
                        file_list = os.listdir(Benign_path)
                        processing(folder_name, "Benign", Benign_Class, Benign_path, file_list)
                elif Class == 'Malignant':
                    Malignant_list = ['Epithelial', 'Non-Epithelial']
                    for Malignant_Class in Malignant_list:
                        Malignant_path = PATH + '/' + folder_name + '/' + Class + '/' + Malignant_Class + '/'
                        file_list = os.listdir(Malignant_path)
                        processing(folder_name, "Malignant", Malignant_Class, Malignant_path, file_list)
                elif Class == 'NP':
                    NP_path = PATH + '/' + folder_name + '/' + Class + '/'
                    file_list = os.listdir(NP_path)
                    processing(folder_name, "NP", "None", NP_path, file_list)
                elif Class == 'Normal':
                    NM_path = PATH + '/' + folder_name + '/' + Class + '/'
                    file_list = os.listdir(NM_path)
                    processing(folder_name, "Normal", "None", NM_path, file_list)
    Before_val = [759, 430, 1974, 1433]
    After_val = [Be, Ma, No, nP]
    data = {'Benign': Be, 'Malignant': Ma, 'Normal': No, 'NP': nP}
    categories = list(data.keys())
    val = list(data.values())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.bar(categories, Before_val, label="Before")
    ax.bar(categories, After_val, label="After")
    ax.set_title("Data size after preprocessing")
    ax.legend()
    plt.show()
    return folder_list


def Data_Augmentation(folder):

    # Folder_list = ["Benign", "Malignant"]
    print(folder)
    sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 9, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 9.0

    sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    if folder == "Normal" or folder == "NP":
        imgs_list = os.listdir("data/processed_data/" + folder +  "/")
        for imgs in imgs_list:
            print(imgs)
            img = cv2.imread("data/processed_data/" + folder + "/" + imgs)
            img_r = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
            img_90 = cv2.rotate(img_r, cv2.ROTATE_90_CLOCKWISE)
            img_180 = cv2.rotate(img_r, cv2.ROTATE_180)
            img_270 = cv2.rotate(img_r, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_r_f = cv2.filter2D(img_r, -1, sharpening_1)
            img_90_f = cv2.filter2D(img_90, -1, sharpening_1)
            img_180_f = cv2.filter2D(img_180, -1, sharpening_1)
            img_270_f = cv2.filter2D(img_270, -1, sharpening_1)

            pred_name_0 = imgs.split(".")[0] + "_90.png"
            pred_name_1 = imgs.split(".")[0] + "_180.png"
            pred_name_2 = imgs.split(".")[0] + "_270.png"
            pred_name_f = imgs.split(".")[0] + "_3filtered.png"
            pred_name_3 = imgs.split(".")[0] + "_90_3filtered.png"
            pred_name_4 = imgs.split(".")[0] + "_180_3filtered.png"
            pred_name_5 = imgs.split(".")[0] + "_270_3filtered.png"
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + imgs, img_r)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_0, img_90)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_1, img_180)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_2, img_270)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_f, img_r_f)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_3, img_90_f)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_4, img_180_f)
            cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_5, img_270_f)
            print("-"*20)
            print(imgs+" Done")
    else:
        class_list = os.listdir("data/processed_data/" + folder + "/")
        for Class in class_list:
            imgs_list = os.listdir("data/processed_data/" + folder + "/" + Class + "/")
            for imgs in imgs_list:
                img = cv2.imread("data/processed_data/"+folder+"/" + Class + "/" + imgs)
                if folder == "Malignant":
                    img_r = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
                    img_90 = cv2.rotate(img_r, cv2.ROTATE_90_CLOCKWISE)
                    img_180 = cv2.rotate(img_r, cv2.ROTATE_180)
                    img_270 = cv2.rotate(img_r, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_r_f = cv2.filter2D(img_r, -1, sharpening_1)
                    img_90_f = cv2.filter2D(img_90, -1, sharpening_1)
                    img_180_f = cv2.filter2D(img_180, -1, sharpening_1)
                    img_270_f = cv2.filter2D(img_270, -1, sharpening_1)
                    img_r_f_5 = cv2.filter2D(img_r, -1, sharpening_2)
                    img_90_f_5 = cv2.filter2D(img_90, -1, sharpening_2)
                    img_180_f_5 = cv2.filter2D(img_180, -1, sharpening_2)
                    img_270_f_5 = cv2.filter2D(img_270, -1, sharpening_2)
                    pred_name_0 = imgs.split(".")[0] + "_90.png"
                    pred_name_1 = imgs.split(".")[0] + "_180.png"
                    pred_name_2 = imgs.split(".")[0] + "_270.png"
                    pred_name_f = imgs.split(".")[0] + "_3filtered.png"
                    pred_name_3 = imgs.split(".")[0] + "_90_3filtered.png"
                    pred_name_4 = imgs.split(".")[0] + "_180_3filtered.png"
                    pred_name_5 = imgs.split(".")[0] + "_270_3filtered.png"
                    pred_name_f_s5 = imgs.split(".")[0] + "_5filtered.png"
                    pred_name_3_s5 = imgs.split(".")[0] + "_90_5filtered.png"
                    pred_name_4_s5 = imgs.split(".")[0] + "_180_5filtered.png"
                    pred_name_5_s5 = imgs.split(".")[0] + "_270_5filtered.png"
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + imgs, img_r)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_0, img_90)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_1, img_180)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_2, img_270)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_f, img_r_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_3, img_90_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_4, img_180_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_5, img_270_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_f_s5, img_r_f_5)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_3_s5, img_90_f_5)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_4_s5, img_180_f_5)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_5_s5, img_270_f_5)

                    print("-" * 20)
                    print(imgs + " Done")
                elif folder == "Benign":
                    img_r = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
                    img_90 = cv2.rotate(img_r, cv2.ROTATE_90_CLOCKWISE)
                    img_180 = cv2.rotate(img_r, cv2.ROTATE_180)
                    img_270 = cv2.rotate(img_r, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_r_f = cv2.filter2D(img_r, -1, sharpening_1)
                    img_90_f = cv2.filter2D(img_90, -1, sharpening_1)
                    img_180_f = cv2.filter2D(img_180, -1, sharpening_1)
                    img_270_f = cv2.filter2D(img_270, -1, sharpening_1)
                    img_r_f_5 = cv2.filter2D(img_r, -1, sharpening_2)
                    img_90_f_5 = cv2.filter2D(img_90, -1, sharpening_2)
                    img_180_f_5 = cv2.filter2D(img_180, -1, sharpening_2)
                    img_270_f_5 = cv2.filter2D(img_270, -1, sharpening_2)
                    pred_name_0 = imgs.split(".")[0] + "_90.png"
                    pred_name_1 = imgs.split(".")[0] + "_180.png"
                    pred_name_2 = imgs.split(".")[0] + "_270.png"
                    pred_name_f = imgs.split(".")[0] + "_3filtered.png"
                    pred_name_3 = imgs.split(".")[0] + "_90_3filtered.png"
                    pred_name_4 = imgs.split(".")[0] + "_180_3filtered.png"
                    pred_name_5 = imgs.split(".")[0] + "_270_3filtered.png"
                    pred_name_f_s5 = imgs.split(".")[0] + "_5filtered.png"
                    pred_name_3_s5 = imgs.split(".")[0] + "_90_5filtered.png"
                    pred_name_4_s5 = imgs.split(".")[0] + "_180_5filtered.png"
                    pred_name_5_s5 = imgs.split(".")[0] + "_270_5filtered.png"
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + imgs, img_r)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_0, img_90)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_1, img_180)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_2, img_270)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_f, img_r_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_3, img_90_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_4, img_180_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_5, img_270_f)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_f_s5, img_r_f_5)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_3_s5, img_90_f_5)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_4_s5, img_180_f_5)
                    cv2.imwrite("data/processed_data_v2/" + folder + "/" + pred_name_5_s5, img_270_f_5)
                    print("-" * 20)
                    print(imgs + " Done")


def Show_data_volume():
    Before_val = [759, 430, 1974, 1433]
    After_val = [Be, Ma, No, nP]
    benign = len(os.listdir("Benign/"))
    malignant = len(os.listdir("Malignant/"))
    normal = len(os.listdir("Normal/"))
    NP = len(os.listdir("NP/"))
    Augmentation = [benign, malignant, normal, NP]
    data = {'Benign': benign, 'Malignant': malignant, 'Normal': normal, 'NP': NP}
    categories = list(data.keys())
    val = list(data.values())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.bar(categories, Augmentation, label="Aug")
    ax.bar(categories, Before_val, label="Before")
    ax.bar(categories, After_val, label="After")
    ax.set_title("Data size after preprocessing")
    ax.legend()
    plt.show()
    print("Benign = ", benign, "Malignant = ", malignant, "Normal = ", normal, "NP = ", NP)


if __name__ == '__main__':
    folder_dir = "data/by hospital/Original"
    read_data_path(folder_dir)
    Folder_list = os.listdir("data/processed_data/")
    for folder in Folder_list:
        Data_Augmentation(folder)
    Show_data_volume()
