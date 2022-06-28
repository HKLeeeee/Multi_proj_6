import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import cv2

# from tensorflow.keras.applications import Densnet
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.data import Dataset

from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from itertools import product
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import iterative_train_test_split
from wandb.keras import WandbCallback
import wandb

params = {
         'learning_rate': 1e-4,
         'epoch': 10,
         'batch_size': 30,
         'drop_out' : 0.5,
         'dense' : 1024
 }


IMAGE_SIZE = 320
MODEL_IMAGE_SIZE = 224

df = pd.read_csv('/home/lab38/Multi_proj_6/data/down_sampled_strawberry.csv')

disease_encoder = LabelEncoder()
disease_encoder.fit(df['disease'])
df['disease'] = disease_encoder.transform(df['disease'])

grow_encoder = LabelEncoder()
grow_encoder.fit(df['grow'])
df['grow'] = grow_encoder.transform(df['grow'])

X_train, X_test, y_train, y_test = train_test_split(df['image'],
                                                    df['disease-grow'],
                                                    stratify=df['disease-grow'],
                                                    test_size=0.2)
train_df = df[df['image'].isin(X_train)]
test_df = df[df['image'].isin(X_test)]

train_gen = ImageDataGenerator(rescale=1./255,
                            rotation_range=20,
                            width_shift_range=0.1, 
                            height_shift_range=0.1,
                            zoom_range=0.2, 
                            horizontal_flip=True, 
                            vertical_flip=True, 
                            fill_mode='nearest')
valid_gen = ImageDataGenerator(rescale= 1. /255.)

train_generator = train_gen.flow_from_dataframe(train_df, 
                                               x_col='image',
                                               y_col=['disease', 'grow'],
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               class_mode='multi_output',
                                               batch_size=params.batch_size)
valid_generator = valid_gen.flow_from_dataframe(test_df,
                                               x_col='image',
                                               y_col=['disease','grow'],
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               class_mode='multi_output',
                                               batch_size=params.batch_size)
base_model= resnet50.ResNet50(
          weights='imagenet',
          include_top=False,
         )


base_model.trainable = False 


input_data = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
x = tf.keras.layers.experimental.preprocessing.Resizing(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)(input_data)
resizing = Model(inputs=input_data, outputs=x, name='resize')


inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = resizing(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(params.drop_out)(x)
backbone_out = layers.Dense(params.dense, activation='softmax')(x)

disease_outputs = layers.Dense(df['disease'].nunique(), activation='softmax',
                        name = 'diease_outputs')(backbone_out)
grow_outputs = layers.Dense(df['grow'].nunique(), activation='softmax',
                    name = 'grow_outputs')(backbone_out)

model = Model(inputs=inputs, 
              outputs=[disease_outputs, grow_outputs],
              name='strawberry')

model.compile(loss={
                  'diease_outputs' : 'sparse_categorical_crossentropy',
                  'grow_outputs' : 'sparse_categorical_crossentropy'
              },
              optimizer=Adam(learning_rate=params.learning_rate),
              metrics=['accuracy'])

model.fit(train_generator,
          validation_data=valid_generator,
          verbose=1,
          epochs=params.epoch,
          steps_per_epoch=len(train_df)//params.batch_size,
         callbacks=[WandbCallback()])
