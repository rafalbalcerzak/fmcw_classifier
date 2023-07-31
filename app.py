import h5py
import numpy as  np
import matplotlib.pyplot as plt
from source import helper
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["image.interpolation"] = 'none'
import tensorflow as tf

from collections import deque
import time
from multiprocessing import Process, Queue


def send(q):
    H5_FILENAME = 'final_dataset'
    loaded_file = h5py.File('data/' + H5_FILENAME + '.h5', 'r')

    keys = list(loaded_file.keys())
    file = 13
    background = loaded_file['background']
    distances = [1, 2, 5, 10, 20, 50]
    max_depth = 50

    frames = loaded_file[keys[file]]
    frames = helper.subtract_background(background, frames)

    frames = loaded_file[keys[file]]
    frames = helper.subtract_background(background, frames)
    spects, y = helper.gen_n_diff_spect(frames, distances=distances)
    spects = np.array(spects)
    spects = spects[:, :50, :]

    spects = [helper.normalize(s) for s in spects]
    spects = np.array(spects)

    frames2 = np.swapaxes(spects, 0, 2)

    start = time.perf_counter()

    for frame in frames2[:1000]:
        q.put(frame)
        print('added frame', flush=True)
        time.sleep(0.01)

def read(q):
    path_model = tf.keras.models.load_model('data/models/model_8.h5')
    classification_model = tf.keras.models.load_model('data/models/class_conv3d_(15,15).h5')
    buffer = deque(maxlen=15)
    obj_class = np.zeros((5, 1))
    bar = np.zeros((51, 1))
    pred = 0
    i = 0
    last_buffer = []
    while True:
        if q.empty(): continue

        while not q.empty():
            buffer.append(q.get().T)

        i += 1
        if len(buffer) == 15:
            frames_buffer = np.array(buffer)
            frames_buffer = np.moveaxis(frames_buffer, 0, 2)

            to_path_predict = np.expand_dims(frames_buffer[:, :, -10:], axis=0)

            position_prediction = path_model.predict(to_path_predict, verbose=0)
            position = np.argmax(position_prediction)
            bar = np.zeros((51, 1))
            bar[position] = 1

            window_size = (15, 15)
            y_dim = window_size[1] // 2
            x_dim = window_size[0]

            window = np.expand_dims(frames_buffer[:, position - y_dim:position + y_dim + 1, :], axis=0)

            if window.shape[2:] == window_size:
                pred = classification_model.predict(window, verbose=0)
                pred = np.argmax(pred)
                obj_class = np.zeros((5, 1))
                obj_class[pred] = 1

            cls = {0: "Bartek",
                   1: "Kuba",
                   2: "Oskar",
                   3: "Rafal",
                   4: "UMO"}

            print(f'{i}\tpos:{position}\tclass:{cls[pred]}', flush=True)


if __name__ == '__main__':
    send_queue = Queue()
    send_p = Process(target=send, args=(send_queue,))
    send_p.daemon = True
    send_p.start()

    read_p = Process(target=read, args=(send_queue,))
    read_p.daemon = True
    read_p.start()
    read_p.join()

