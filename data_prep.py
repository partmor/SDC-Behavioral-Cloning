import numpy as np
import pandas as pd
import cv2
import pickle

def import_data_from_sample_drive(data_path, use_lateral=False): 

    csv_file_path = data_path + 'driving_log.csv'
    img_folder_path = data_path + 'IMG/'

    img_cols = ['center_img', 'left_img', 'right_img']
    status_cols = ['steering', 'throttle', 'brake', 'speed']

    df = pd.read_csv(csv_file_path, header=None)
    df.columns = img_cols + status_cols

    for img_col in img_cols:
        df[img_col] = df[img_col].str.split(r'/|\\').apply(lambda x: x[-1])
        df[img_col] = img_folder_path + df[img_col]


    images_center = list()
    images_left = list()
    images_right = list()

    for ix, row in df.iterrows():
        images_center.append(cv2.imread(row['center_img']))
        if use_lateral:
            images_left.append(cv2.imread(row['left_img']))
            images_right.append(cv2.imread(row['right_img']))

    steering_angles = df['steering'].values

    X_train = np.array(images_center + images_left + images_right)
    if use_lateral:
        y_train = np.concatenate([steering_angles]*3)
    else:
        y_train = steering_angles.copy()
    
    return X_train, y_train

def import_data_from_n_sample_drives(data_path_list, load_cached=False, save_data_to_pickle=True):
    
    if load_cached:
        X_train = pickle.load(open('X_train.p', 'rb'))
        y_train = pickle.load(open('y_train.p', 'rb'))
        
        return X_train, y_train
        
    x_list = list()
    y_list = list()
    
    for data_path in data_path_list:
        x, y = import_data_from_sample_drive(data_path)
        x_list.append(x)
        y_list.append(y)
    
    X_train = np.concatenate(x_list)
    y_train = np.concatenate(y_list)
    
    if save_data_to_pickle:
        pickle.dump(X_train, open('X_train.p', 'wb'))
        pickle.dump(y_train, open('y_train.p', 'wb'))
    
    return X_train, y_train

def flipping_augmentation(X_train, y_train):
    
    flipped_list = list()

    for i in range(X_train.shape[0]):
        flipped_list.append(np.fliplr(X_train[i])[np.newaxis])
    X_train_flipped = np.concatenate(flipped_list)
    y_train_flipped = - y_train

    x_res = np.concatenate([X_train, X_train_flipped])
    y_res = np.concatenate([y_train, y_train_flipped])

    return x_res, y_res