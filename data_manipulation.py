from skimage.io import imread
from skimage.exposure import adjust_gamma
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re
import cv2

def out_of_pipeline_preprocessing(img):
    output_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return output_img


def get_balanced_dataset_indices(dataset, num_bins, thresh='mean'):
    selection_list = list()

    # the data is balanced based on the absolute value steering angle
    y = dataset['steering']
    y_abs = np.abs(y)

    # binning of the measurements
    hist, bins = np.histogram(y_abs, bins=num_bins)
    bin_idx = np.digitize(np.abs(y), bins=bins)
    if thresh=='mean':
        thresh = hist.mean()
    
    # for each bin pick a random subsample of a maximum size given by `thresh`
    for a, b in zip(np.roll(bins, shift=1)[1:], bins[1:]):
        idx_in_bin = y[(y_abs > a) & (y_abs <= b)].index
        if len(idx_in_bin) > 0:
            subset = np.random.choice(idx_in_bin, size=min(int(thresh), len(idx_in_bin)), replace=False)
            selection_list.append(subset)
    
    # return the indices of the balanced data
    selected_idx = np.concatenate(selection_list)
    return selected_idx


def load_samples_log(log_path, balancing_num_bins=50, balancing_thresh='mean'):
    df = pd.read_csv(log_path, header=None).iloc[1:]

    # set column names
    img_cols = ['center_img', 'left_img', 'right_img']
    status_cols = ['steering', 'throttle', 'brake', 'speed']
    df.columns = img_cols + status_cols
    df = df[img_cols + ['steering']]

    # data balancing
    selected_idx = get_balanced_dataset_indices(df, num_bins=balancing_num_bins, thresh=balancing_thresh)
    df = df.loc[selected_idx]
    return df


def apply_random_section_shading(img):
    h, w = img.shape[0], img.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        img[i, :c, :] = (img[i, :c, :] * .5).astype(np.uint8)
    return img


def apply_random_gamma_adjust(img, low, high):
    gamma = np.random.random_sample() * (high - low) + low
    return adjust_gamma(img, gamma=gamma)


def batch_generator(
    samples, 
    samples_folder, 
    batch_size, 
    use_lat=True, use_flip=True, use_shade=True, use_gamma=True,
    corr_angle=0.25
    ):

    num_samples = samples.shape[0]
    # loop forever so the generator never terminates
    while True: 
        samples = samples.loc[np.random.permutation(samples.index)]
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : (offset + batch_size)]

            cameras = {
                0: 'left_img',
                1: 'center_img',
                2: 'right_img'
            }

            angle_correction = {
                0: corr_angle,
                1: 0.0,
                2: - corr_angle
            }

            img_list = list()
            angle_list = list()

            for ix, batch_sample in batch_samples.iterrows():

                # randomly select between left/center/right cameras and apply corresponding correction
                cam = np.random.randint(3) if use_lat else 1
                # use regex to be able to read paths with normal slash and backslash
                img_fname = '%sIMG/%s' % (samples_folder,  re.compile(r'/|\\').split(batch_sample[cameras[cam]])[-1])
                image = imread(img_fname)
                angle = batch_sample['steering'] + angle_correction[cam]
                
                # randomly decide whether to flip or not the image
                if np.random.choice([True, False]):
                    image = np.fliplr(image)
                    angle = - angle

                # apply random brightness variations through gamma adjustment
                if use_gamma:
                    image = apply_random_gamma_adjust(image, low=0.4, high=1.5)

                # apply random shading to a section of the image
                if use_shade:
                    image = apply_random_section_shading(image)

                img_list.append(out_of_pipeline_preprocessing(image))
                angle_list.append(angle)


            X_train = np.array(img_list)
            y_train = np.array(angle_list)
            yield shuffle(X_train, y_train)