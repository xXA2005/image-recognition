from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import os


dataset_dir = "./dataset/"
image_dimensions = (128,128)
batch_size = 32
epoches = 50
learning_rate = 0.001



train = ImageDataGenerator(rescale=1/255)


train_dataset = train.flow_from_directory(dataset_dir,
                                          target_size=image_dimensions,
                                          class_mode="categorical",
                                          batch_size=batch_size,
                                          subset="training")

validation_dataset = train.flow_from_directory(dataset_dir,
                                          target_size=image_dimensions,
                                          class_mode="categorical",
                                          batch_size=batch_size,
                                          subset="validation")
print(train_dataset.class_indices)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_dimensions[0], image_dimensions[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(os.listdir(dataset_dir)), activation='softmax')
])




model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation
model.fit(train_dataset, epochs=epoches, validation_data=validation_dataset)

# Save the trained model
model.save('model.h5')

