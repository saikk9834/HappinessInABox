import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = './'

#Select the directory where the training and validation data is present
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics = ['acc'])

train = ImageDataGenerator( rescale = 1.0/255. )
test  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train.flow_from_directory(train_dir, batch_size=20, class_mode='binary',target_size = (150,150))

validation_generator =  test.flow_from_directory(validation_dir, batch_size=20, class_mode  = 'binary', target_size = (150, 150))
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_steps=50,
                              verbose=2,
                              use_multiprocessing=False)

#Once the data is trained based on the given dataset, use the model to predict for an image of our choice

import tkinter as tk
from tkinter.filedialog import askopenfilename

tk.Tk().withdraw()
print("Select the image to be checked")
file = askopenfilename()

from PIL import Image

img = Image.open(file)

import smtplib
import os
import pytesseract
import cv2

#After the file is selected, try predicting it using the model that we have created. If the image is from the 'empty' directory, or basically it gives '0' as output,
#Then select the bottom part of the image that has the name of the item and the upper half which contains the details about the image
#Now apply OCR(Optimal Character Recognition) on the images using pytesseract to extract strings from the cropped images
if (history.predict(img) == 0):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w,h = img1.size
    crop_part1 = (0,h/2,w,h)
    img_1 = img1.crop(crop_part1)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, img_1)
    text1 = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    crop_part2 = (0,h,w,h/2)
    img_2 = img1.crop(crop_part2)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, img_2)
    text2 = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    content = text1 + text2
    #Once the data is extracted from the images, create a mail SMTP server to send email to the concerned person containing the concerned data
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.starttls() 
    mail.login("sender_email_id", "sender_email_id_password") 
    mail.sendmail("sender_email_id", "receiver_email_id", content) 
    mail.quit() 
