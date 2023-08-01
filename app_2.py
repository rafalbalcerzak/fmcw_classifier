import h5py
import numpy as  np
from source import helper
import tensorflow as tf
import pygame
from collections import deque
import time
from multiprocessing import Process, Queue
import pygame.freetype

def send(q,d, file):
    while True:
        H5_FILENAME = 'final_dataset'
        loaded_file = h5py.File('data/' + H5_FILENAME + '.h5', 'r')
        keys = list(loaded_file.keys())
        background = loaded_file['background']
        distances = [1, 2, 5, 10, 20, 50]
        max_depth = 50

        frames = loaded_file[keys[file]]
        frames = helper.subtract_background(background, frames)
        spects, y = helper.gen_n_diff_spect(frames, distances=distances)
        spects = np.array(spects)
        spects = spects[:, :50, :]

        spects = [helper.normalize(s) for s in spects]
        spects = np.array(spects)

        frames2 = np.swapaxes(spects, 0, 2)

        start = time.perf_counter()

        for frame in frames2:
            q.put(frame)
            d.put(frame)
            # print('added frame', flush=True)
            time.sleep(0.01)

def read(q, s):
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

            # print(f'{i}\tpos:{position}\tclass:{cls[pred]}', flush=True)
            s.put([position, cls[pred]])


if __name__ == '__main__':
    send_queue = Queue()
    display_queue = Queue()
    label_queue = Queue()

    H5_FILENAME = 'final_dataset'
    loaded_file = h5py.File('data/' + H5_FILENAME + '.h5', 'r')

    keys = list(loaded_file.keys())
    _ = [print(f'{i}: {k}') for i, k in enumerate(keys)]
    file = input('Typ file number: ')
    file = int(file)

    send_p = Process(target=send, args=(send_queue, display_queue, file))
    send_p.daemon = True
    send_p.start()

    read_p = Process(target=read, args=(send_queue, label_queue))
    read_p.daemon = True
    read_p.start()
    # read_p.join()

    # display_p = Process(target=display, args=(send_queue, display_queue))
    # display_p.daemon = True
    # display_p.start()
    # display_p.join()

    pygame.init()
    window = pygame.display.set_mode((700, 500))
    pygame.display.set_caption("FMCW")
    # GAME_FONT = pygame.freetype.Font("Comic Sans MS", 24)
    GAME_FONT = pygame.font.SysFont('Jet Brains Mono', 40)
    GAME_FONT_SMALL = pygame.font.SysFont('Jet Brains Mono', 20)
    run = True

    bar_bg = (255, 255, 255)
    bar_ind = (255, 100, 100)

    arr = np.zeros((15, 50))
    buffer = deque(maxlen=100)
    pos, cls = 0,None
    cls_buff = deque(maxlen=5)

    while run:
        if not send_p.is_alive():
            print('end')
            run = False

        pygame.time.delay(2)

        while not display_queue.empty():
            data = display_queue.get()
            data = np.sum(data, axis=1)/6
            buffer.append(data)
            # buffer.append(display_queue.get()[:,0])

        while not label_queue.empty():
            pos, cls = label_queue.get()


        img = np.array(buffer)
        img = img*255

        # img = np.squeeze(img, axis=-1)

        img = img.astype('int')
        img = np.expand_dims(img, axis=-1)
        img = np.kron(img, np.ones((3, 10, 3)))
        surf = pygame.surfarray.make_surface(img)

        for event in pygame.event.get():
            if event.type == pygame.K_q:
                run = False
            if event.type == pygame.QUIT:
                run = False


        window.fill((0, 0, 0))
        window.blit(surf, (300-img.shape[0], 0))
        cls_buff.append(cls)

        dom_cls = max(set(cls_buff), key=cls_buff.count)
        if dom_cls != None:

            text_surface = GAME_FONT.render(str(dom_cls), False, (255, 255, 255))
            # window.blit(text_surface, (320, 250-20))
            window.blit(text_surface, (320, pos*10-20))
            pygame.draw.rect(window, bar_bg, pygame.Rect(300,0,10,500))

            pygame.draw.rect(window, bar_ind, pygame.Rect(300, pos*10, 10, 10))

            val = str(round(pos * 0.15, 2)) + 'm'
            dist = GAME_FONT.render(val, False, (255, 255, 255))
            window.blit(dist, (520, pos * 10-20))

        else:
            text_surface = GAME_FONT_SMALL.render('Waiting for radar', False, (255, 255, 255))
            window.blit(text_surface, (320, window.get_height()//2-10))

        pygame.display.update()

