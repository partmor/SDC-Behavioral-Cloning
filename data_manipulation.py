from skimage.io import imread
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re

def load_samples_log(log_path):
    df = pd.read_csv(log_path, header=None).iloc[1:]

    img_cols = ['center_img', 'left_img', 'right_img']
    status_cols = ['steering', 'throttle', 'brake', 'speed']

    df.columns = img_cols + status_cols
    df = df[img_cols + ['steering']]
    # naive downsampling
    df = df[np.abs(df['steering']) > 1/100]
    return df

def batch_generator(samples, samples_folder, batch_size, use_lat=True, use_flip=True, corr_angle=0.25):
    num_samples = samples.shape[0]
    # loop forever so the generator never terminates
    while True: 
        # shuffle samples
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
                img_fname = '%sIMG/%s' % (samples_folder,  re.compile(r'/|\\').split(batch_sample[cameras[cam]])[-1])
                image = imread(img_fname)
                angle = batch_sample['steering'] + angle_correction[cam]
                
                # randomly decide whether to flip or not the image
                if np.random.choice([True, False]):
                    image = np.fliplr(image)
                    angle = - angle

                img_list.append(image)
                angle_list.append(angle)


            X_train = np.array(img_list)
            y_train = np.array(angle_list)
            yield shuffle(X_train, y_train)