import time
import pika
from multiprocessing import Queue, Process, Value
from multiprocessing.shared_memory import ShareableList
import functools
from collections import deque
import numpy as np
from ctypes import c_int

data_queue = Queue()
prediction = Value(c_int, 1)
to_print_queue = Queue()


def on_message(ch, method, properties, body, args):
    input = args[0]
    prediction = args[1]
    to_print = args[2]
    arr = body.split()
    print(len(arr))
    print(arr)
    input.put(body)
    to_print.put([body, prediction.value])
    print(prediction.value)

def prediction_model(*args):
    q = args[0]
    prediction = args[1]
    buffer = deque(maxlen=10)

    while True:
        if q.qsize() > 0:
            while q.qsize() > 0:
                buffer.append(q.get())
        else:
            print('end of data')
            buffer.clear()
            continue

        # prediction
        time.sleep(0.1)
        if len(buffer) < 10:
            print(f'to_short {len(buffer)}')
            continue

        buffer_decoded = [int(b.decode('utf-8')) for b in buffer]

        print(len(buffer), buffer_decoded)
        prediction.value = buffer_decoded[-1]


def rabbit(q, pred, to_print):
    print('Starting rabbit')
    connection_parameters = pika.ConnectionParameters('localhost')
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel()
    on_message_callback = functools.partial(on_message, args=(q, pred, to_print))
    channel.basic_consume(queue='letterbox', auto_ack=True, on_message_callback=on_message_callback)
    channel.start_consuming()


if __name__ == "__main__":
    printer = Process(target=prediction_model, args=(data_queue, prediction))
    printer.daemon = True
    printer.start()

    rabbit_t = Process(target=rabbit, args=(data_queue, prediction, to_print_queue))
    rabbit_t.daemon = True
    rabbit_t.start()

    rabbit_t.join()
