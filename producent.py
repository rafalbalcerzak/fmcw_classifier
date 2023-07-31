# docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.9-management
import pika
import h5py
import numpy as  np
import matplotlib.pyplot as plt
from source import helper
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["image.interpolation"] = 'none'
import tensorflow as tf
import time

connection_parameters = pika.ConnectionParameters('localhost')

connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()
channel.queue_declare(queue='letterbox')

H5_FILENAME = 'final_dataset'
loaded_file = h5py.File('data/'+ H5_FILENAME + '.h5','r')
path_model = tf.keras.models.load_model('data/models/model_8.h5')
classification_model = tf.keras.models.load_model('data/models/class_conv3d_(15,15).h5')
file = 15
background = loaded_file['background']
distances = [1,2,5,10,20,50]
max_depth = 50
keys = list(loaded_file.keys())
frames = loaded_file[keys[file]]
frames = helper.subtract_background(background, frames)
background = loaded_file['background']
distances = [1, 2, 5, 10, 20, 50]
max_depth = 50

frames = loaded_file[keys[file]]
frames = helper.subtract_background(background, frames)

frames = loaded_file[keys[file]]
frames = helper.subtract_background(background, frames)
spects,y = helper.gen_n_diff_spect(frames, distances=distances)
spects = np.array(spects)
spects = spects[:,:50,:]

spects = [helper.normalize(s) for s in spects]
spects = np.array(spects)
frames2 = np.swapaxes(spects, 0, 2)
print(frames2.shape)

last_time = time.perf_counter()
for frame in frames2[1000:2000]:
    channel.basic_publish(exchange='', routing_key='letterbox', body=str(frame))
    # time.sleep(0.01)


connection.close()