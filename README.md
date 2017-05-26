# **CarND: Behavioral Cloning**  [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---
[//]: # (Image References)

[unbalanced_hist]: ./examples/unbalanced_hist.png
[balanced_hist]: ./examples/balanced_hist.png
[random_shade]: ./examples/random_shade.png 
[gamma]: ./examples/gamma.png 
[raw_frame]: ./examples/raw_frame.png 
[cropped_frame]: ./examples/cropped_frame.png 


The goal of this project is to train a convolutional neural network (**CNN**) based model, using an **end-to-end** approach: raw pixels from a front-facing camera are mapped to the steering commands for a self-driving car.

The data of *good driving behaviour* is collected driving a car around a track in a [simulator](https://github.com/udacity/self-driving-car-sim). A CNN built with [Keras](https://keras.io/) is trained with the latter data to predict steering angles from images. Eventually this model is assessed by testing it successfully drives around the track without leaving the road.

## Project structure

The project includes the following files:

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `data_manipulation.py`                    | Contains methods for data loading, balancing, and augmentation.                  |
| `model.py`                   | Definition of the model architecture and training process.
| `model.h5`                   | Contains the trained model outputed in `model.py`  |
| `drive.py`                   | In autonomous mode, it communicates with the driving simulator guiding the car based on real time predictions provided by `model.h5`. |

The model is built, trained and saved executing: 
```sh
python model.py
```
and once the simulator app is set to *Autonomous Mode*, the prediction stream is established with the following instruction:
```sh
python drive.py model.h5
```

## Data collection

The driving simulator saves frames from the car's point of view corresponding to three front-facing cameras, together with the *state* of the vehicle: throttle, speed and steering angle.

The training data for the CNN is collected from **track 1**. The car is driven *safelly* around the track, always focusing on keeping the car in the center of the lane.

To help the model generalize better, the following actions are performed during data collection:
+ Completing laps in both **clockwise** and **counter-clockwise** directions, accomplishing a higher balance of left/right curves.
+ **Recovery** scenarios are recorded: taking the car intentionally out of the path and recording the *just* the recovery maneuver. This will increase the model's robustness against perturbations.

As mentioned, the aim is to train the model to be able to map raw image inputs to steering angles.

## Data balancing

Despite of taking care of driving the car around the track in both directions and recording recovery maneuvers, the dataset is still highly unbalanced:

![unbalanced_hist]

According to the figure above, if trained with the dataset as is, the model would tend to drive in a straight line, something that we want to avoid.

Balancing is performed by sublampling the dataset; the samples are binned based on the **absolute value** of the steering angle, and bin size is restricted to be lower than an amount `thresh`:

```python
def balance_dataset(dataset, num_bins, thresh='mean'):
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
        if idx_in_bin.size > 0:
            subset = np.random.choice(idx_in_bin, size=min(int(thresh), idx_in_bin.size), replace=False)
            selection_list.append(subset)
    
    # return the indices of the balanced data
    selected_idx = np.concatenate(selection_list)
    return selected_idx
```

Using this method, the original unbalanced dataset with **8535** samples is downsampled to **2874**, now with the following distribution:

![balanced_hist]

Bear in mind that balancing is performed across absolute values, since applying horizontal flip during augmentation will mitigate potential small unbalances between left/right turns.

## Data augmentation

Tha data balancing from the latter step yielded a 2.8 k sample dataset that is clearly insufficient to train a model that could generalize well. However, the following augmentation techniques altogether enable to extend the dataset by an order of magnitude:
+ **Left and right cameras**: each collected sample includes images taken from 3 camera positions: left, center and right. During the autonomous driving stage (test) the only input will be the center camera frame. However, the lateral cameras can still be used for training (selecting position randomly) by applying a correction `corr_angle` to the associated steering angle. This increases samples by a factor of 3.
```python
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
...
...
# randomly select between left/center/right cameras and apply corresponding correction
cam = np.random.randint(3) if use_lat else 1
# use regex to be able to read paths with normal slash and backslash
img_fname = '%sIMG/%s' % (samples_folder,  re.compile(r'/|\\').split(batch_sample[cameras[cam]])[-1])
image = imread(img_fname)
angle = batch_sample['steering'] + angle_correction[cam]
```
+ **Horizontal flip**: frames are horizontally flipped at random, together with a change in the sign of the steering angle. This increases samples by a factor of **2**.
```python
# randomly decide whether to flip or not the image
if np.random.choice([True, False]):
    image = np.fliplr(image)
    angle = - angle
```
+ **Random shadow**: a random slice shadowing of the image is performed, by decreasing brightness of a random frame slice, hoping to make the model robust against shadows, marks, and texture variation on the road.
```python
def apply_random_section_shading(img):
    h, w = img.shape[0], img.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        img[i, :c, :] = (img[i, :c, :] * .5).astype(np.uint8)
    return img
```

![random_shade]

+ **Random brightness**: the brightness of the image is altered using a range of random values for gamma adjustment. The objective is also increase robustness against lighting conditions and tarmack tonalities.
```python
def apply_random_gamma_adjust(img, low, high):
    gamma = np.random.random_sample() * (high - low) + low
    return adjust_gamma(img, gamma=gamma)
```

![gamma]

These augmentation techniques are encapsulated in a pipeline to work as a **generator**, allowing to augment data real-time, without having to worry about running out of memory for instance.

Following this pipeline, the original dataset can be effectively augmented at least by a factor of 6, while also enhancing the robustness of the model by pertubing the images.

All the augmentation steps except horizontal flip are disabled to generate the validation set.

## Preprocessing

Training, validation, and eventually test data run through a common pipeline that comprises:
+ **YUV colorspace transformation**: the images are converted from RGB to YUV colorspace, after the augmentation pipeline in the training phase, and during collection (within `drive.py`) in the testing phase. The YUV colorspace was used in [NVIDIA's](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) end-to-end deep learning for self-driving cars project. This election is supported by the added robustness that this colorspace provides for detecting features in images for automotive applications, since varying lighting conditions are isolated effectively by the *luma* Y channel, from the color information enclosed in the U and V channels. The usage of this colorspace could somehow reduce the effectiveness of the transformations from the augmentation pipeline where shading and brightness conditions are altered.

+ **Image cropping**: as seen in the image bellow, the raw frames collected by the camera have regions that provide little information of interest and just add noise; that is the landscape above the horizon of the lane, and the bonnet of the car. Cropping the top 50 rows and bottom 25 rows of pixels of the original image can help the model focus on extracting the important lane features. This step is performed within the model with a Keras `Cropping2D` layer:
```python
model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))
```

![raw_frame]

![cropped_frame]

+ **Scaling**: scaling the input helps the convergence of the optimization algorithm during the training phase. A simple [0,1] scaling of the input pixels followed by a mean substraction is performed with a Keras `Lambda` layer:
```python
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
```
