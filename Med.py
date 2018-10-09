from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#
from keras.models import model_from_json
#import numpy

#initialize the CNN
classifier = Sequential()

#step 1 - Convolution2D
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))

#step 2 -MaxPooling2D
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3 -flattering
classifier.add(Flatten())

#step 4 -full connection
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#part 2 -fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/DEADPOOL/Documents/final_project/my_imple/dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('/home/DEADPOOL/Documents/final_project/my_imple/dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')



from IPython.display import display
from PIL import Image

classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=10,validation_data=test_set,validation_steps=800)
#save the model in JSON
model_json = classifier.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

classifier.save_weights("model.h5")
print('model saved')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/DEADPOOL/Documents/final_project/my_imple/random.jpg',target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image ,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >=0.5:
    prediction = 'bicycle'
else:
    prediction = 'car'
print(prediction)
