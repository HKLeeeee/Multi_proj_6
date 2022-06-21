import paho.mqtt.client as client
import paho.mqtt.publish as publisher
from threading import Thread
import cv2
import numpy as np
import urllib.request as req
import tensorflow as tf
import torch

class awsImageSub:
    # mqtt 초기 설정 코드(클라이언트, 콜백함수 설정)
    def __init__(self):
        self.myclient = client.Client()
        self.myclient.on_connect = self.on_connect
        self.myclient.on_message = self.on_message
        #이 곳에 변수설정이나 여려 설정 해 주세요
        self.STRAWBERRY = 'Strawberry'
        self.LETTUCE = 'Lettuce'
        self.ROSEMARY = 'Rosemary'
        self.GERANIUM = 'Geranium'
        self.MultiModelInputSize = 320
        self.strawberry_dict = {
            'disease' : ['정상', '잿빛곰팡이병', '흰가루병', '해충'],
            'grow': ['1단계', '2단계', '3단계', '4단계', '5단계']
            }
        self.lettuce_dict = {
            'disease': ['정상', '균핵병', '노균병'],
            'grow': ['1단계', '2단계']
            }
        self.rosemary_dict = {
            'disease': ['정상', '흰가루병', '점무늬병', '해충']
            }
        self.geranium_dict = {
            'disease': ['정상', '갈색무늬병', '잿빛곰팡이병', '해충']
            }

        self.code_to_str = {
                self.STRAWBERRY : self.strawberry_dict, 
                self.LETTUCE: self.lettuce_dict, 
                self.ROSEMARY: self.rosemary_dict, 
                self.GERANIUM: self.geranium_dict
        }
    
    # url을 이용해서 이미지를 불러옴
    # url이 아니라 파일경로라면 함수 수정이 필요함.
    def read_image_from_url(self, url, input_size = None) :
        resp = req.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if input_size is not None :
            image = cv2.resize(image, (input_size, input_size))
        return image / 255.

    # 생성된 모델은 이파일이 올라가는 EC2 환경의 tensorflow의 버전과 모델을 학습한 버전이 동일해함
    # EC2상의 tensorflow version == 2.4.1
    def load_plant_model(self, plant : str):
        if plant ==  self.STRAWBERRY :
            model = tf.keras.models.load_model('./strawberry_multi_output.h5')
        elif plant == self.LETTUCE :
            model = tf.keras.models.load_model('./lettuce_multi_output.h5')
        elif plant == self.ROSEMARY :
            model = tf.keras.models.load_model('모델저장경로')
        elif plant == self.GERANIUM :
            model = tf.keras.models.load_model('모델저장경로')
        return model

    def inference(self, url, plant:str) :
        # load model
        model = self.load_plant_model(plant)
        
        if plant == self.STRAWBERRY or plant == self.LETTUCE :
            # multi model은 리사이즈 필요
            image = self.read_image_from_url(url, MultiModelInputSize)
            image = np.array([image])
            disease, grow = model.predict(image)
            disease = np.argmax(disease[0])
            grow = np.argmax(grow[0])    
            return {'disease' : code_to_str[plant]['disease'][disease],
                   'grow' : code_to_str[plant]['grow'][grow]}
    
        elif plant == ROSEMARY :
            image = read_image_from_url(url)
            image = np.array([image])
            disease = model.predict(image)
            disease = np.argmax(disease[0])
            return {'disease': code_to_str[plant]['disease'][disease]}
    
        elif plant == GERANIUM :
            image = read_image_from_url(url)
            image = np.array([image])
            disease = model.predict(image)
            disease = np.argmax(disease[0])
            return {'disease': code_to_str[plant]['disease'][disease]}


        
    # mqtt를 스레드를 사용해서 연결하기 위한 코드
    def mymqtt_connect(self):
        try:
            print("브로커 연결 시작하기")
            self.myclient.connect("35.182.237.235", 1883, 60)
            mythreadobj = Thread(target=self.myclient.loop_forever)
            mythreadobj.start()
        except KeyboardInterrupt:
            pass
        finally:
            print("종료")

    # 브로커에 연결이 된 경우 동작하는 콜백함수
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("연결 완료")
            client.subscribe("AI/directory") # 토픽 구독
        else:
            print("연결실패")

    # 구독한 토픽이 설정된 메시지가 들어오면 동작하는 콜백함수
    def on_message(self, client, userdata, message):
        try:
            subPayload = message.payload.split(':')
            directory = message.payload[0] #받은 메시지에 실려있는 페이로드(경로를) 변수에 저장
            print(directory) # S3 버킷의 경로 출력
            #여기부터
            pred = inference(direcotry, subPayload[1]) # 결과는 여기에 집어넣어 주세요
            
            if (subPayload[1] == self.STRAWBERRY or subPayload[1] == self.LETTUCE):
                # 받은 경로가 딸기, 상추 인 경우
                if(subPayload[2] == "lvDisChk"):
                    # 생육상태 판단 + 질병판단 같이 보내기
                    result = pred["grow"]+"/"+pred["disease"]
                elif(subPayload[2] == "disOnlyChk"):
                    # 질병만 보내기
                    result = "7/"+pred["disease"]
            else: #나머지
                result = "7/질병코드"  # 질병만 보내기
            publisher.single("iot/aiToIotValue",result,hostname="35.182.237.235")
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