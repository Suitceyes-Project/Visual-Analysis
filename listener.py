import json
import numpy as np
import requests
import time
import uuid
import os
from threading import Lock, Thread
import datetime
import cv2
import analyzer
import paho.mqtt.client as mqtt
from PIL import Image
import imagehash


print("Suitceyes Visual analysis")

#The storage location of the uploaded images and jsons
STORAGE_URL = '/Path_to_location/'
storage_path = '/Path_to_location/'

frame_count = 0
mean_analysis_interval = 0.7
analysis_interval_sum = 0
threads_on = 0
T_queue = []
first_img = True

#MQTT protocol for communication
Connected = False
#Requires broker
client = mqtt.Client()
client.username_pw_set()
client.tls_set()

lock = Lock()

def on_connect(client, userdata, flags, rc):
    print('Connected')
    global Connected                #Use global variable
    Connected = True 

def on_connect_send(client, userdata, flags, rc):
    print('Connected')
    global Connected_send                #Use global variable
    Connected_send = True 

def on_disconnect(client, userdata, rc):
    print('Disconnected')
    client.loop_stop()
    
def on_message(client, userdata, message):
    time.sleep(1)
    global T_queue
    # make new thread and push to FIFO queue.
    T_queue += [Thread(target=message_handler, args=(str(message.payload.decode("utf-8")), len([each for each in T_queue if each.is_alive()])))]
    T_queue[-1].start()
    print('\nNew message is being processed by thread {}.'.format(T_queue[-1].ident))
    #print("received message =",str(message.payload.decode("utf-8")))

def read_local_image(image_path):
    img_np_bgr = cv2.imread(image_path)
    img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    return img_np_bgr, img_np_rgb

def read_depth_reg(image_path):
    depth_reg = cv2.imread(image_path, -1)
    return depth_reg

def rotate_image_90(image_np):
    return np.rot90(image_np)

def rotate_image_180(image_np):
    return rotate_image_90(rotate_image_90(image_np))

def send_results_to_KB(json_data):
    #Message with json location that contains the Visual analysis 
    message_id = "VA_"+uuid.uuid4().hex
    data_json_filename = message_id+'.json'
    with open(os.path.join(storage_path, data_json_filename), mode='w') as file:
        json.dump(json_data, file)
    link = STORAGE_URL+data_json_filename

    topic_va_kb = {'header':{'sender':'VA', 'recipients':['KBS'], 'timestamp':'', 'message_id':''}, 'body':{'data':''}}
    topic_va_kb['header']['timestamp'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[0:-3]
    topic_va_kb['header']['message_id'] = message_id
    topic_va_kb['body']['data'] = link
    client.publish('VA_KBS_channel', json.dumps(topic_va_kb),qos=0)
    print('\tSending message with id:{} to VA_KBS_channel'.format(message_id))

def rt_on_message(sender, channel, message):
    global T_queue
    # make new thread and push to FIFO queue.
    T_queue += [Thread(target=message_handler, args=(message, len([each for each in T_queue if each.is_alive()])))]
    T_queue[-1].start()
    print('\nNew message is being processed by thread {}.'.format(T_queue[-1].ident))
    
def message_handler(message, num_of_waiting_threads):
    global mean_analysis_interval
    global analysis_interval_sum
    global frame_count
    global threads_on
    global first_img
    global previous
    
    threads_on += 1
    
    
    capture_timestamp_str = message.split(sep='-')[0]
    try:
        capture_timestamp = datetime.datetime.strptime(capture_timestamp_str, "%d_%m_%y_%H_%M_%S_%f")
    except:
        print("Invalid request, skipping.")
        threads_on -= 1
        return -1
    
    upload_interval = datetime.datetime.utcnow() - capture_timestamp
    upload_interval = upload_interval.total_seconds()
    processing_time_estimator = upload_interval+threads_on*mean_analysis_interval
    

    #lock here so incoming messages will not be skipped
    lock.acquire()
    start = time.time()
    
    message_flags = message.split(sep='-')[1:]
    rgb_filename = 'rgb_'+capture_timestamp_str+'.jpg'
    depth_reg_filename = 'depth_reg_'+capture_timestamp_str+'.png'
    
    try:
        image_np_bgr, image_np_rgb = read_local_image(os.path.join(storage_path, rgb_filename))
        img = Image.open(os.path.join(storage_path, rgb_filename))
    except:
        print("Could not read images, skipping.")
        lock.release()
        threads_on -= 1
        return -1
    
    # possible flags:
        # -nodepth: there is no depth provided
        # -conf=x: confidence drop is x
        # -cl=se_demo, or -cl=se_oid : choose class pool
    
    if 'nodepth' in message_flags:
        depth_reg = None
    else:
        try:            
            #read depth_reg
            depth_reg = read_depth_reg(os.path.join(storage_path, depth_reg_filename))

            print(image_np_rgb.shape, depth_reg.shape)
        except:
            print("Could not read depth image, skipping that.")

            depth_reg = None
        
    try:
        conf_t = float([each for each in message_flags if 'conf' in each][0].split(sep='=')[1])
    except:
        conf_t = 0.5
        
    try:
        cl_p = [each for each in message_flags if 'cl' in each][0].split(sep='=')[1]
    except:
        cl_p = 'se_demo'
    #Compare incoming image with the previous one to check the degree of similarity
    if  first_img == True:
        similar = -1

        previous = img
        first_img = False
    else:
        hash_now = imagehash.average_hash(img)
        hash_prev = imagehash.average_hash(previous)
        similar = abs(hash_prev - hash_now)
       
        previous = img

    hash_now = imagehash.average_hash(img)
    analysis_flags = ["save_data_json_locally"]
    result_dict, json_data = analyzer.process_image(similar,image_np_rgb, depth_reg,
                                                    confidence_drop = conf_t,
                                                    class_pool = cl_p,
                                                    file_name=capture_timestamp_str,
                                                    timestamp=capture_timestamp,
                                                    flags=analysis_flags)
    end = time.time()
    frame_count += 1
    analysis_interval = end-start
    analysis_interval_sum += analysis_interval
    mean_analysis_interval = analysis_interval_sum/frame_count
    full_process_interval = datetime.datetime.utcnow() - capture_timestamp
    full_process_interval = full_process_interval.total_seconds()
    #Analysis metrics
    print('\timage upload and processing estimated {} seconds'.format(processing_time_estimator))
    print('\timage upload and processing took {} seconds'.format(full_process_interval))
    print('\timage upload took {} seconds'.format(upload_interval))
    print('\timage processing took {} seconds'.format(analysis_interval))
    print('\tsum of upload and processing is {} seconds'.format(upload_interval+analysis_interval))
    print('\tactive threads are {}. mean analysis time is {}'.format(threads_on, mean_analysis_interval))
    fps = frame_count/analysis_interval_sum
    print('VA runs at {} fps'.format(fps))
    lock.release()
    
    send_results_to_KB(json_data)
    analyzer.visualize_results(image_np_bgr, capture_timestamp_str, result_dict)
    
    threads_on -= 1
    return 0

print('Performing a test analysis...')
image_np_bgr, image_np_rgb = read_local_image('/Path_to_location/rgb_06_08_19_16_25_47_051563.jpg')
capture_timestamp = datetime.datetime.utcnow()
capture_timestamp_str = capture_timestamp.strftime("%d_%m_%y_%H_%M_%S_%f")
analyzer.process_image(-1,image_np_rgb, None,
                       file_name=capture_timestamp_str+'_va-controler-test', 
                       timestamp=capture_timestamp)
print('Done.')


print("subscribing ")
client.subscribe("VA_RS_channel",qos=0)#subscribe

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message=on_message
client.connect('mqtt.ably.io', port=8883, keepalive=15)

client.loop_start()
while Connected != True:    #Wait for connection
    time.sleep(0.1)

while True:
    #listen for incoming messages
    continue
