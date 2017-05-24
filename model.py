from data_manipulation import load_samples_log, batch_generator
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout


DATA_FOLDER = 'recorded_data/merged_data/'
LOG_FNAME = 'driving_log.csv'

BATCH_SIZE = 128
LAT_ANGLE_CORR = 0.27

samples_df = load_samples_log(DATA_FOLDER + LOG_FNAME)
train_df, valid_df = train_test_split(samples_df, test_size=0.2, random_state=2017)

train_generator = batch_generator(
    train_df, samples_folder=DATA_FOLDER, batch_size=BATCH_SIZE,
    corr_angle=LAT_ANGLE_CORR
)
valid_generator = batch_generator(
    valid_df, samples_folder=DATA_FOLDER, batch_size=BATCH_SIZE, 
    use_lat=False
)

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

def build_Nvidia_Regularized(model):
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(1))

# select model architecture
build_model = build_Nvidia_Regularized

model = Sequential()
# pre-processing
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# set up architecture
build_model(model)

# train the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    steps_per_epoch=6*train_df.shape[0]/BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=2*valid_df.shape[0]/BATCH_SIZE,
    epochs=5
)

# save the model
model.save('model.h5')