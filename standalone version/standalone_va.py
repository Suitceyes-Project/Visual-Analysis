import json
import numpy as np
import time
import picamera
import os
import datetime
import paho.mqtt.client as mqtt
import uuid
import scene
import analyzer_light
import cv2


storage_path = '/Path_to_location'
def on_connect(client, userdata, flags, rc):
    print('Connected')
    global Connected                #Use global variable
    Connected = True 

def on_disconnect(client, userdata, rc):
    print('Disconnected')
    client.loop_stop()

#VAriable initialization
scene_fin = "scene_unknown"
sc_score = 0

print('========== Ably MQTT Script ==========\n\n')
Connected = False
client = mqtt.Client()
#Each client uses separate Api key
client.username_pw_set()
client.tls_set()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.connect('mqtt.ably.io', port=8883, keepalive=60)


#RPI camera image  resolution and frame rate
camera = picamera.PiCamera()
camera.resolution = (640,480)
camera.framerate = 30
time.sleep(2)
first_time = True

#Analysis results forwarding via MQTT
def send_results_to_KB(json_data):
    message_id = "VA_"+uuid.uuid4().hex
    data_json_filename = message_id+'.json'
    with open(os.path.join(storage_path, data_json_filename), mode='w') as file:
        json.dump(json_data, file)
    link = storage_path+'/'+data_json_filename

    topic_va_kb = {'header':{'sender':'VA', 'recipients':['KBS'], 'timestamp':'', 'message_id':''}, 'body':{'data':''}}
    topic_va_kb['header']['timestamp'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[0:-3]
    topic_va_kb['header']['message_id'] = message_id
    topic_va_kb['body']['data'] = json_data
    client.publish('SVA_KBS_channel', json.dumps(topic_va_kb))
    
def read_local_image(image_path):
    img_np_bgr = cv2.imread(image_path)
    img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    return img_np_bgr, img_np_rgb

while True:
    start = time.time()
    capture_timestamp = datetime.datetime.utcnow()
    capture_timestamp_str = capture_timestamp.strftime("%d_%m_%y_%H_%M_%S_%f")
    camera.capture(capture_timestamp_str + ".jpg", use_video_port=True)
    end = time.time()
    interval1 = end-start
    print('capture interval is {}'.format(interval1))
    

    start = time.time()
    time.sleep(1-interval1)
    image_np_bgr, image_np_rgb = read_local_image("Path_to_location" + capture_timestamp_str + ".jpg")
    link = (capture_timestamp_str+ "_VA")


    #If the laplacian of the image is below the threshold then the image is deemed blurry
    if  cv2.Laplacian(image_np_bgr, cv2.CV_64F).var() > 50:
        if first_time == True:
           print("First run")
           #During the first run the process is slower
           result_dict, json_data = analyzer_light.process_image(scene_fin,sc_score,capture_timestamp,image_np_bgr,None,link)
           first_time = False
        else:
           scene_fin,sc_score = scene.process_scene(image_np_bgr)
           result_dict, json_data = analyzer_light.process_image(scene_fin,sc_score,capture_timestamp,image_np_bgr,None,link)
           send_results_to_KB(json_data)
        

        end = time.time()
        interval2 = end-start

        os.remove('{}.jpg'.format(capture_timestamp_str))
        print('analyze interval is {}'.format(interval2))
    else:
        print("Blurry image")
