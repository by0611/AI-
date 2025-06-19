from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from IPython.display import display, HTML
%matplotlib inline

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
#tf.enable_eager_execution()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print('Tensorflow version: ',tf.__version__)
print('Is GPU available: %s' % str(tf.test.is_gpu_available()))
print('Is Eager Execution enabled?: %s' % str(tf.executing_eagerly()))

import boto3
import os

s3 = boto3.client('s3',
endpoint_url = 'https://s3.wasabisys.com',
aws_access_key_id = 'VM2WCNG36U812Y1NGCT3',
aws_secret_access_key='g3Dqovv3IYlIFDZyNWONXZSU5yhGZvWhKOJrBQRI')

def download_files(filenames, save_dir):
    for i, filename in enumerate(filenames):
         print('Downloading %d: %s' % (i, filename)) 
         download_file(filename, save_dir)

def download_file(filename, save_dir):
    full_filename = os.path.join(save_dir,filename)
    if os.path.exists(full_filename):
        print('\tAlready have: %s' % full_filename)
        return
    s3.download_file('curaedlhw',filename,full_filename)
    print('\tCOMPLETE: %s saved to %s' % (filename, save_dir))

FILELIST = ['chest_xray_train_val.tar.gz']

IS_ONEPANEL = False
NEED_DOWNLOAD = False
if IS_ONEPANEL:
    DATA_DIR='/onepanel/input/datasets/curae/chestx-ray8-mini/1'
else:
    import os
    DATA_DIR = 'data'
    if NEED_DOWNLOAD:
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        download_files(FILELIST,DATA_DIR)

os.environ["DATADIR"] = DATA_DIR

meta_file = os.path.join(DATA_DIR,'Data_Entry_2017_v2020.csv')
meta_df = pd.read_csv(meta_file)

meta_df.head()

meta_df.describe()

class_map = {'Cardiomegaly': 0,
             'Emphysema': 1,
             'Effusion': 2,
             'Hernia': 3,
             'Infiltration': 4,
             'Mass': 5,
             'Nodule': 6,
             'Atelectasis': 7,
             'Pneumothorax' : 8,
             'Pleural_Thickening': 9,
             'Pneumonia' : 10,
             'Fibrosis' : 11,
             'Edema': 12,
             'Consolidation': 13}
class_list = [''] * 14
for class_name, class_index in class_map.items():
  class_list[class_index] = class_name

def labels_to_one_hot(label_str):
    label = np.zeros(len(class_map), dtype=int)
    for disease in label_str.split('|'):
        disease = disease.strip()
        if disease in class_map:
            label[class_map[disease]] = 1
    return label

labels = np.array([labels_to_one_hot(s) for s in meta_df['Finding Labels']])

def convert_to_label_string(arr):
    label = []
    for i, val in enumerate(arr):
        if val == 1:
            label.append(class_list[i])
    return '|'.join(label)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(meta_df['Image Index'].values, labels, test_size=0.2, random_state=42)

print("Size of train set: %d, test set: %d" % (len(train_y),len(test_y)))

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ChestXraySequence(Sequence):
    def __init__(self, x_set, y_set, batch_size,folder):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.folder = folder

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        count = len(batch_x)

        img_list = []
        for file_name in batch_x:
            img = imread(os.path.join(self.folder, file_name), as_gray=True)
            img_resized = resize(img, (224, 224), mode='reflect', anti_aliasing=True)
            img_list.append(img_resized)

        img_batch = np.array(img_list, dtype=np.float32).reshape(count, 224, 224, 1)
        img_batch = np.repeat(img_batch, 3, axis=3)

        return preprocess_input(img_batch), np.array(batch_y)

BATCH_SIZE=32
image_folder = os.path.join(DATA_DIR, 'images')

train_seq = ChestXraySequence(x_set=train_x,
                              y_set=train_y,
                              batch_size=BATCH_SIZE,
                              folder=image_folder)

test_seq = ChestXraySequence(x_set=test_x,
                              y_set=test_y,
                              batch_size=BATCH_SIZE,
                              folder=image_folder)

x, y = test_seq[0]

x.shape, y.shape

sample_img, sample_y = x[:10], y[:10]

fig, ax = plt.subplots(5,2, figsize=(20,20))
for i, (img, label) in enumerate(zip(sample_img, sample_y)):
    ax[i//2,i%2].title.set_text(convert_to_label_string(label))
    ax[i//2,i%2].imshow((img+1)/2)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, History
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg'
)
for layer in pretrained_model.layers:
  layer.trainable = False

def build_model(base):
    predictions = tf.keras.layers.Dense(len(class_list), activation='sigmoid')(base.output)
    return Model(inputs=base.input,outputs=predictions)
    
model = build_model(pretrained_model)

model.summary()

LEARNING_RATE = 0.0001
EPOCHS = 10

model.compile(
  optimizer= tf.keras.optimizers.Adam(LEARNING_RATE),
  loss='binary_crossentropy',
  metrics=['acc'])

save_best = tf.keras.callbacks.ModelCheckpoint('best_model',save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3,min_lr=LEARNING_RATE/1000)
tb_cb = tf.keras.callbacks.TensorBoard(log_dir='/onepanel/output/model4')
hist = tf.keras.callbacks.History()
callbacks = [save_best, early_stopping, reduce_lr, hist, tb_cb]

history = model.fit(
    train_seq,
    steps_per_epoch=len(train_seq),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=test_seq,
    validation_steps=len(test_seq),
    verbose=1
)

import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history.get('acc') or history.history.get('accuracy')
    val_acc = history.history.get('val_acc') or history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    epochs = range(1, len(acc) + 1)

    # 準確率
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 損失值
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 用法（訓練完成後執行）
plot_training_history(history)
