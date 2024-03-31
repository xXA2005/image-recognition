from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import os

dataset_dir = "./dataset/"
image_dimensions = (128, 128)
batch_size = 16
epochs = 50
learning_rate = 0.001

train = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    dataset_dir,
    target_size=image_dimensions,
    class_mode="categorical",
    batch_size=batch_size,
    subset="training"
)

print(train_dataset.class_indices)

model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(units=128, activation='relu'),
    Dense(units=len(os.listdir(dataset_dir)), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy')

model.fit(train_dataset, epochs=epochs)

model.save('model.keras')
