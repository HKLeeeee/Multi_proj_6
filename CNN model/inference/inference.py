import paho.mqtt.client as client
import paho.mqtt.publish as publisher
from threading import Thread
import cv2
import numpy as np
import urllib.request as req
import tensorflow as tf
import torch
import time
import os

class awsImageSub:
    # mqtt 초기 설정 코드(클라이언트, 콜백함수 설정)
    def __init__(self):
        self.myclient = client.Client()
        self.myclient.on_connect = self.on_connect
        self.myclient.on_message = self.on_message
        self.IpAddress = "3.96.178.99"
        self.STRAWBERRY = 'Strawberry'
        self.LETTUCE = 'Lettuce'
        self.ROSEMARY = 'Rosemary'
        self.GERANIUM = 'Geranium'
        self.MultiModelInputSize = 320
        self.strawberry_dict = {
            'disease' : ['정상', '잿빛곰팡이병', '흰가루병', '해충'],
            'grow': ['1', '2', '3', '4', '5']
            }
        self.lettuce_dict = {
            'disease': ['정상', '균핵병', '노균병'],
            'grow': ['1', '2']
            }
        self.rosemary_dict = {
            'disease': ['정상', '점무늬병', '해충', '흰가루병']
            }
        self.geranium_dict = {
            'disease': ['정상', '잿빛곰팡이병', '갈색무늬병', '해충']
            }

        self.code_to_str = {
                self.STRAWBERRY : self.strawberry_dict, 
                self.LETTUCE: self.lettuce_dict, 
                self.ROSEMARY: self.rosemary_dict, 
                self.GERANIUM: self.geranium_dict
        }
        
        
        self.strawberry_model =  tf.keras.models.load_model('./strawberry_multi_output.h5')
        # self.lettuce_model = tf.keras.models.load_model('./lettuce_multi_output.h5')
        self.rosemary_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_rosemary.pt')
        self.geranium_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_geranium.pt')
    
    def read_image_from_dir(self, img_dir, input_size = None) :
        image = cv2.imread(img_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if input_size is not None :
            image = cv2.resize(image, (input_size, input_size))
        return image / 255.

    def inference(self, img_dir, plant:str) :
        if plant == self.STRAWBERRY :
            # multi model은 리사이즈 필요
            model = self.strawberry_model
            image = self.read_image_from_dir(img_dir, self.MultiModelInputSize)
            image = np.array([image])
            disease, grow = model.predict(image)
            disease = np.argmax(disease[0])
            grow = np.argmax(grow[0])    
            return {'disease' : self.code_to_str[plant]['disease'][disease],
                   'grow' : self.code_to_str[plant]['grow'][grow]}
        
        elif plant == self.LETTUCE :
            model = self.lettuce_model
            image = self.read_image_from_dir(img_dir, self.MultiModelInputSize)
            image = np.array([image])
            disease, grow = model.predict(image)
            disease = np.argmax(disease[0])
            grow = np.argmax(grow[0])    
            return {'disease' : self.code_to_str[plant]['disease'][disease],
                   'grow' : self.code_to_str[plant]['grow'][grow]}
    
        elif plant == self.ROSEMARY:
            model = self.rosemary_model
            model.names[0] = 0 # 'Rosemary'
            model.names[1] = 1 # 'Rosemary Leaf Spot'
            model.names[2] = 2 # 'Rosemary Pest Damage'
            model.names[3] = 3 # 'Rosemary Powdery Mildew'
            model.conf = 0.40
            model.multi_label = False
            model.max_det = 1
            image = [img_dir]
            disease = model(image, size=416)
            disease = np.argmax(disease[-1])
            return {'disease': self.code_to_str[plant]['disease'][disease]}
    
        elif plant == self.GERANIUM:
            model = self.geranium_model
            model.names[0] = 0 # 'Geranium'
            model.names[1] = 1 # 'Geranium Gray Mold'
            model.names[2] = 2 # 'Geranium Leaf Spot'
            model.names[3] = 3 # 'Geranium Pest Damage'
            model.conf = 0.40
            model.multi_label = False
            model.max_det = 1
            image = [img_dir]
            disease = model(image, size=256)
            return {'disease': self.code_to_str[plant]['disease'][disease]}
            # 만약 'image 1/2: 740x416 1 3' 이런식으로 리턴된다면, disease[-1]로 대체해보자


    # mqtt를 스레드를 사용해서 연결하기 위한 코드
    def mymqtt_connect(self):
        try:
            print("브로커 연결 시작하기")
            self.myclient.connect(self.IpAddress, 1883, 60)
            mythreadobj = Thread(target=self.myclient.loop_forever)
            mythreadobj.start()
        except KeyboardInterrupt:
            pass
        finally:
            print("종료")

    def createFolder(self, directory):
        try: 
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory. " + directory)
            
    # 브로커에 연결이 된 경우 동작하는 콜백함수
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("연결 완료")
            client.subscribe("AI/directory") # 토픽 구독
            client.subscribe("iot/awslevelDisease/#")
            client.subscribe("iot/awsperiodic/#")
        else:
            print("연결실패")

    # 구독한 토픽이 설정된 메시지가 들어오면 동작하는 콜백함수
    def on_message(self, client, userdata, message):
        try:
            topic = message.topic
            if topic == "AI/directory" :
                subPayload = message.payload.decode("utf-8").split(':')
                directory = subPayload[0] #받은 메시지에 실려있는 페이로드(경로를) 변수에 저장
                print(directory) # S3 버킷의 경로 출력
                #여기부터
                pred = self.inference(directory, subPayload[1]) # 결과는 여기에 집어넣어 주세요
                print('Inference Done!')
                
                result=''
                if (subPayload[1] == self.STRAWBERRY or subPayload[1] == self.LETTUCE):
                    # 받은 경로가 딸기, 상추 인 경우
                    if(subPayload[2] == "lvDisChk"):
                        # 생육상태 판단 + 질병판단 같이 보내기
                        result = pred["grow"]+"/"+pred["disease"]
                    elif(subPayload[2] == "disOnlyChk"):
                        # 질병만 보내기
                        result = "7/"+pred["disease"]
                else: #나머지
                    result = "7/"+pred["disease"]  # 질병만 보내기
                print(result)
                publisher.single("iot/aiToIotValue",result,hostname=self.IpAddress)
            else :
                topic = topic.split('/')
                directory = "/home/ec2-user/image/" + topic[2] # 들어온 식물명으로 폴더를 만들기
                self.createFolder(directory)
                currenttime = time.strftime("%Y%m%d-%H%M%S")
                f = open(directory + "/" + str(currenttime) + ".jpg", "wb")
                myval = bytearray(message.payload)
                f.write(myval)
                f.close()

                if topic[1] == "awslevelDisease":
                    publisher.single("AI/directory",
                                 directory + "/" + str(currenttime) + ".jpg" +
                                 ":" + topic[2] +
                                 ":" + "lvDisChk",
                                 hostname=self.IpAddress)
                elif topic[1] == "awsperiodic":
                    publisher.single("AI/directory", directory + "/" + str(currenttime) + ".jpg" +
                                     ":" + topic[2] +
                                     ":" + "disOnlyChk",hostname=self.IpAddress)
        except KeyboardInterrupt:
            pass
        finally:
            pass


if __name__ == "__main__":
    try:
        awsImageSubFirst = awsImageSub()
        awsImageSubFirst.mymqtt_connect()
    except KeyboardInterrupt:
        pass
    finally:
        pass
