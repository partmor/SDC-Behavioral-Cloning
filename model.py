from data_prep import import_data_from_n_sample_drives, flipping_augmentation, downsample_data

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout

data_paths = [
    #'recorded_data/track1_3laps_ccw/',
    #'recorded_data/track1_3laps_cw/',
    'recorded_data/sample_data/'
]

X_train, y_train = import_data_from_n_sample_drives(
    data_paths, 
    load_cached=True, save_data_to_pickle=True,
    use_lateral=True, lat_st_corr=0.25
    )
X_train, y_train = downsample_data(X_train, y_train)
X_train, y_train = flipping_augmentation(X_train, y_train)

def build_LeNet_Default(model):
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

def build_Nvidia_Default(model):
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))

model = Sequential()

# pre-processing
model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# set up architecture
build_Nvidia_Default(model)

# train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

# save the model
model.save('model.h5')