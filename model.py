from data_manipulation import load_samples_log, batch_generator
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout


data_folder = 'recorded_data/sample_data/'

batch_size = 128
samples_df = load_samples_log(data_folder + 'driving_log.csv')
train_df, valid_df = train_test_split(samples_df, test_size=0.2)

train_generator = batch_generator(
    train_df, samples_folder=data_folder, batch_size=batch_size
)
valid_generator = batch_generator(
    valid_df, samples_folder=data_folder, batch_size=batch_size, use_lat=False, use_flip=False
)

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

# select model architecture
build_model = build_Nvidia_Default

model = Sequential()
# pre-processing
model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# set up architecture
build_model(model)

# train the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    steps_per_epoch=train_df.shape[0]/batch_size,
    validation_data=valid_generator,
    validation_steps=valid_df.shape[0]/batch_size,
    epochs=2
)

# save the model
model.save('model.h5')