from re import T
from traceback import print_tb
from django.shortcuts import redirect, render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import threading
import cv2, time, dlib # conda install -c conda-forge dlib
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import requests, json
from django.contrib.auth.decorators import login_required
from PIL import Image
from django.http import HttpResponse, FileResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
import pygame

actions = ['walking', 'sitting', 'reposing']
seq_length = 30
colors = [(245,117,16), (117,245,16), (16,117,245)]
model = load_model('models/model_2.h5')

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
cap = cv2.VideoCapture('img/input_save_2.mp4')
global time_list
time_list = []
url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
seq = []
action_seq = []
reposing_warning_time = 10
reposing_danger_time = 30
# img_warning = 'img/warning.png'
# img_danger = 'img/danger.png'
warning = cv2.imread('img/warning.png')
danger = cv2.imread('img/danger.png')
warning = cv2.resize(warning,(150, 70))
danger = cv2.resize(danger, (150, 70))
global i
i = 0

# ----------------------------------------------------------------

# 얼굴 인식 -------------------------------------------------------
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
koo = cv2.imread('img/koo.jpg')
koo = cv2.resize(koo,(150, 150))
byeon = cv2.imread('img/byeon1.jpg')
byeon = cv2.resize(byeon,(150, 150))
descs = np.load('img/descs.npy',allow_pickle=True)[()] # ,allow_pickle=True
global face_find
face_find = list()
# def encode_face(img):
#     dets = detector(img, 1)

#     if len(dets) == 0:
#         return np.empty(0)

#     for k, d in enumerate(dets):
#         shape = sp(img, d)
#         face_descriptor = facerec.compute_face_descriptor(img, shape)

#         return np.array(face_descriptor)


# def prob_viz(conf, this_action, input_frame, colors):
#     img = input_frame.copy()
#     for num, prob in enumerate(conf):
#         cv2.rectangle(img, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#         cv2.putText(img, this_action[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


# Create your views here.
# https://blog.miguelgrinberg.com/post/video-streaming-with-flask/page/8

def index(request):
    context = {}

    return render(request, "index.html", context)

def get_frame(cap):
    global time_list
    global i
    global warning_sound
    global danger_sound
    global face_find
    ret, img = cap.read()
    img0 = img.copy()
    # img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img_bgr = cv2.copyMakeBorder(img, top=padding_size, bottom=padding_size, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    # 얼굴인식 -----------------------------------------------------------------------------------------------

    dets = detector(img, 1)

    if len(dets) == 0:
        dets = np.empty(0)

    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptor = np.array(face_descriptor)
    for k, d in enumerate(dets):
        shape = sp(img_rgb, d)
        face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)
        last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

        for name, saved_desc in descs.items():
            dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)
            if dist < last_found['dist']:
                last_found = {'name': name, 'dist': dist, 'color': (0,255,0)}

                if last_found['name'] == 'Koo Tae Wan':
                    rows, cols, channels = koo.shape
                    # 삽입하고자 하는 이미지의 위치의 로고크기만큼의 이미지 컷팅
                    # roi = img_bgr[10:10+rows , 260:260+cols]
                    roi = img[10:10+rows , 480:480+cols]

                    # 로고 이미지를 GRAYSCALE로 변환
                    logoGray = cv2.cvtColor(koo,cv2.COLOR_BGR2GRAY)

                    # 임계값 설정을 통한 mask 이미지 생성하기
                    # threshold 라는 함수는 GRAYSCALE 만 사용가능함.
                    # (대상이미지, 기준치, 적용값, 스타일)
                    # 해당 cv2.THRESH_BINARY 는 이미지내의 픽셀값이 기준치 이상인 값들은
                    # 모두 255로 부여함. 즉 픽셀값이 100이상이면 흰색, 100미만이면 검정색으로 표시
                    # 변환된 이미지는 mask에 담김
                    ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)

                    # 임계값 설정을 한 이미지를 흑백 반전시킴
                    # mask_inv = cv2.bitwise_not(mask)

                    # 위에서 자른 로고크기의 커피사진 영역에 mask에 할당된 이미지의
                    # 0이 아닌 부분만 roi 와 roi 이미지를 AND 연산합니다.
                    # 즉 커피이미지에서 로고크기만큼의 영역에 로고의 모양만 0값이 부여됩니다.
                    im2_bg = cv2.bitwise_and(roi, roi, mask=mask)

                    # 로고이미지에서 로고모양을 제외하고 다 0값을 가지게됩니다.
                    im1_fg = cv2.bitwise_and(koo,koo,mask=mask)

                    # 로고크기만큼의 영역의 이미지에 로고이미지를 연산합니다.
                    dst = cv2.addWeighted(im2_bg, 0, im1_fg, 1, 0)
                    # dst = cv2.add(im2_bg, im1_fg)

                    # 커피의 원본이미지에 컷팅된 영역에 로고가 삽입된 이미지를 붙여넣습니다.
                    # img_bgr[10:10+rows , 260:260+cols] = dst
                    img[10:10+rows , 480:480+cols] = dst
                    # cv2.rectangle(img, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
                    # cv2.putText(img, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
                    face_find.append('Koo Tae Wan')
                
                elif last_found['name'] == 'Byeon Ui Hyeok':
                    rows, cols, channels = byeon.shape
                    # 삽입하고자 하는 이미지의 위치의 로고크기만큼의 이미지 컷팅
                    # roi = img_bgr[10:10+rows , 260:260+cols]
                    roi = img[10:10+rows , 480:480+cols]

                    # 로고 이미지를 GRAYSCALE로 변환
                    logoGray = cv2.cvtColor(byeon,cv2.COLOR_BGR2GRAY)

                    # 임계값 설정을 통한 mask 이미지 생성하기
                    # threshold 라는 함수는 GRAYSCALE 만 사용가능함.
                    # (대상이미지, 기준치, 적용값, 스타일)
                    # 해당 cv2.THRESH_BINARY 는 이미지내의 픽셀값이 기준치 이상인 값들은
                    # 모두 255로 부여함. 즉 픽셀값이 100이상이면 흰색, 100미만이면 검정색으로 표시
                    # 변환된 이미지는 mask에 담김
                    ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)

                    # 임계값 설정을 한 이미지를 흑백 반전시킴
                    # mask_inv = cv2.bitwise_not(mask)

                    # 위에서 자른 로고크기의 커피사진 영역에 mask에 할당된 이미지의
                    # 0이 아닌 부분만 roi 와 roi 이미지를 AND 연산합니다.
                    # 즉 커피이미지에서 로고크기만큼의 영역에 로고의 모양만 0값이 부여됩니다.
                    im2_bg = cv2.bitwise_and(roi, roi, mask=mask)

                    # 로고이미지에서 로고모양을 제외하고 다 0값을 가지게됩니다.
                    im1_fg = cv2.bitwise_and(byeon,byeon,mask=mask)

                    # 로고크기만큼의 영역의 이미지에 로고이미지를 연산합니다.
                    dst = cv2.addWeighted(im2_bg, 0, im1_fg, 1, 0)
                    # dst = cv2.add(im2_bg, im1_fg)

                    # 커피의 원본이미지에 컷팅된 영역에 로고가 삽입된 이미지를 붙여넣습니다.
                    # img_bgr[10:10+rows , 260:260+cols] = dst
                    img[10:10+rows , 480:480+cols] = dst
                    face_find.append('Byeon Ui Hyeok')
                    
                else:
                    pass
        cv2.rectangle(img, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
        cv2.putText(img, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
    
# 뼈다귀 --------------------------------------------------------------------------------
    poselist = list()
    if result.pose_landmarks is not None:
        poselist.append(result.pose_landmarks)
        for res in poselist:
            joint = np.zeros((33, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,11,13,0,12,14,11,23,25,12,24,26], :3] # Parent joint
            v2 = joint[[11,13,15,12,14,16,23,25,27,24,26,28], :3] # Child joint
            v = v2 - v1 # [11, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,1,0,3,4,4,3,6,7,9,10],:], 
                v[[1,2,6,6,4,5,9,9,7,8,10,11],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

            if len(seq) < seq_length:
                continue
            
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            reslt = model.predict(input_data)[0]
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)
            
            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                if this_action == 'reposing':

                    time_list.append(time.time())
                    if time.time() - time_list[0] > reposing_warning_time and reposing_danger_time > time.time() - time_list[0]:
                        # playsound('C:\Final_Project_Visual\danger.mp3', False)
                        pygame.init() # pygame을 진행할 때 꼭 최적화를 해줘야합니다.
                        global warning_sound
                        warning_sound = pygame.mixer.Sound('C:/Final_Project_Visual/dwarning.mp3')
                        
                        warning_sound.play() # 정의한 소리 한번 재생
                        rows, cols, channels = warning.shape
                        # 삽입하고자 하는 이미지의 위치의 로고크기만큼의 이미지 컷팅
                        roi = img[10:10+rows , 260:260+cols]

                        # 로고 이미지를 GRAYSCALE로 변환
                        logoGray = cv2.cvtColor(warning,cv2.COLOR_BGR2GRAY)

                        # 임계값 설정을 통한 mask 이미지 생성하기
                        # threshold 라는 함수는 GRAYSCALE 만 사용가능함.
                        # (대상이미지, 기준치, 적용값, 스타일)
                        # 해당 cv2.THRESH_BINARY 는 이미지내의 픽셀값이 기준치 이상인 값들은
                        # 모두 255로 부여함. 즉 픽셀값이 100이상이면 흰색, 100미만이면 검정색으로 표시
                        # 변환된 이미지는 mask에 담김
                        ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)

                        # 임계값 설정을 한 이미지를 흑백 반전시킴
                        # mask_inv = cv2.bitwise_not(mask)

                        # 위에서 자른 로고크기의 커피사진 영역에 mask에 할당된 이미지의
                        # 0이 아닌 부분만 roi 와 roi 이미지를 AND 연산합니다.
                        # 즉 커피이미지에서 로고크기만큼의 영역에 로고의 모양만 0값이 부여됩니다.
                        im2_bg = cv2.bitwise_and(roi, roi, mask=mask)

                        # 로고이미지에서 로고모양을 제외하고 다 0값을 가지게됩니다.
                        im1_fg = cv2.bitwise_and(warning,warning,mask=mask)

                        # 로고크기만큼의 영역의 이미지에 로고이미지를 연산합니다.
                        dst = cv2.add(im2_bg, im1_fg)

                        # 커피의 원본이미지에 컷팅된 영역에 로고가 삽입된 이미지를 붙여넣습니다.
                        img[10:10+rows , 260:260+cols] = dst
                        # cv2.putText(img, 'warning!!!!', (240, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                        i += 1
                        pygame.QUIT

                        # 카카오톡 
                        if i < 2:
                            if 'Byeon Ui Hyeok' in face_find or 'Koo Tae Wan' in face_find:    
                                with open("json/kakao_code.json","r") as fp:
                                    tokens = json.load(fp)

                                url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

                                # kapi.kakao.com/v2/api/talk/memo/default/send 

                                headers={
                                    "Authorization" : "Bearer " + tokens["access_token"]
                                }

                                data={
                                    "template_object": json.dumps({
                                        "object_type":"text",
                                        "text":face_find[0]+"님이 누워 일어나지 않습니다 *경고*(10초)",
                                        "link":{
                                            "web_url":"www.naver.com"
                                        }
                                    })
                                }
                                response = requests.post(url, headers=headers, data=data)
                                response.status_code
                            else:
                                with open("json/kakao_code.json","r") as fp:
                                    tokens = json.load(fp)

                                url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

                                # kapi.kakao.com/v2/api/talk/memo/default/send 

                                headers={
                                    "Authorization" : "Bearer " + tokens["access_token"]
                                }

                                data={
                                    "template_object": json.dumps({
                                        "object_type":"text",
                                        "text":"확인이 되지 않는 사람이 누워 일어나지 않습니다 *경고*(10초)", # a+
                                        "link":{
                                            "web_url":"www.naver.com"
                                        }
                                    })
                                }
                                
                                response = requests.post(url, headers=headers, data=data)
                                response.status_code
                            warning_sound.stop()
                        
                    elif int(time.time()) - int(time_list[0]) == reposing_danger_time:
                        i = 0
                        pygame.QUIT
                        warning_sound.stop()
                    elif time.time() - time_list[0] > reposing_danger_time:
                        # playsound("C:\Final_Project_Visual\danger.mp3", False)
                        pygame.init() # pygame을 진행할 때 꼭 최적화를 해줘야합니다.
                        
                                
                        danger_sound = pygame.mixer.Sound('C:/Final_Project_Visual/danger.mp3')
                        
                        danger_sound.play()
                        
                        rows, cols, channels = danger.shape
                        # 삽입하고자 하는 이미지의 위치의 로고크기만큼의 이미지 컷팅
                        roi = img[10:10+rows , 260:260+cols]

                        # 로고 이미지를 GRAYSCALE로 변환
                        logoGray = cv2.cvtColor(danger,cv2.COLOR_BGR2GRAY)

                        # 임계값 설정을 통한 mask 이미지 생성하기
                        # threshold 라는 함수는 GRAYSCALE 만 사용가능함.
                        # (대상이미지, 기준치, 적용값, 스타일)
                        # 해당 cv2.THRESH_BINARY 는 이미지내의 픽셀값이 기준치 이상인 값들은
                        # 모두 255로 부여함. 즉 픽셀값이 100이상이면 흰색, 100미만이면 검정색으로 표시
                        # 변환된 이미지는 mask에 담김
                        ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)

                        # 임계값 설정을 한 이미지를 흑백 반전시킴
                        # mask_inv = cv2.bitwise_not(mask)

                        # 위에서 자른 로고크기의 커피사진 영역에 mask에 할당된 이미지의
                        # 0이 아닌 부분만 roi 와 roi 이미지를 AND 연산합니다.
                        # 즉 커피이미지에서 로고크기만큼의 영역에 로고의 모양만 0값이 부여됩니다.
                        im2_bg = cv2.bitwise_and(roi, roi, mask=mask)

                        # 로고이미지에서 로고모양을 제외하고 다 0값을 가지게됩니다.
                        im1_fg = cv2.bitwise_and(danger,danger,mask=mask)

                        # 로고크기만큼의 영역의 이미지에 로고이미지를 연산합니다.
                        # dst = cv2.add(im2_bg, im1_fg)
                        dst = cv2.addWeighted(im2_bg, 0, im1_fg, 1, 0)

                        # 커피의 원본이미지에 컷팅된 영역에 로고가 삽입된 이미지를 붙여넣습니다.
                        img[10:10+rows , 260:260+cols] = dst
                        i += 1

                        if i < 2:
                        # 카카오톡
                            # if len(face_find) != 0:
                            if 'Byeon Ui Hyeok' in face_find or 'Koo Tae Wan' in face_find:
                                with open("json/kakao_code.json","r") as fp:
                                    tokens = json.load(fp)

                                url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

                                # kapi.kakao.com/v2/api/talk/memo/default/send 

                                headers={
                                    "Authorization" : "Bearer " + tokens["access_token"]
                                }

                                data={
                                    "template_object": json.dumps({
                                        "object_type":"text",
                                        "text":face_find[0]+"님이 누워 일어나지 않습니다 *위험*(60초)", # a+
                                        "link":{
                                            "web_url":"www.naver.com"
                                        }
                                    })
                                }
                                
                                response = requests.post(url, headers=headers, data=data)
                                response.status_code
                            else:
                                with open("json/kakao_code.json","r") as fp:
                                    tokens = json.load(fp)

                                url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

                                # kapi.kakao.com/v2/api/talk/memo/default/send 

                                headers={
                                    "Authorization" : "Bearer " + tokens["access_token"]
                                }

                                data={
                                    "template_object": json.dumps({
                                        "object_type":"text",
                                        "text":"확인이 되지 않는 사람이 누워 일어나지 않습니다 *위험*(60초)",
                                        "link":{
                                            "web_url":"www.naver.com"
                                        }
                                    })
                                }
                                
                                
                                response = requests.post(url, headers=headers, data=data)
                                response.status_code
                            danger_sound.stop()
                            
                        else:
                            
                            pygame.QUIT
                    else:
                        i=0
                        pygame.QUIT
                else:
                    time_list = []
                    # pygame.quit()
                    pygame.QUIT
            # prob_viz(reslt, actions, img, colors)
            img = img.copy()
            for num, prob in enumerate(reslt):
                cv2.rectangle(img, (0,345+num*40), (int(prob*100), 380+num*40), colors[num], -1)
                cv2.putText(img, actions[num], (0, 370+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.putText(img, f'{this_action.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
# --------------------------------------------------------------------------------------------------------------



    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

def gen(get_frame):
    while cap.isOpened():
        frame = get_frame(cap)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# back_path = 'img/data.jpg'
# def back_get_frame(back_path):
#     # back_img = cv2.imread(back_path)
#     # back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
#     # back_img = cv2.cvtColor(back_img, cv2.COLOR_RGB2BGR)
#     # _, jpeg = cv2.imencode('.jpg', back_img)
#     # return jpeg.tobytes()
#     ret, img = cv2.VideoCapture(back_path)
#     _, jpeg = cv2.imencode('.jpg', img)
#     return jpeg.tobytes()

# def back_gen(back_get_frame):
#     while True:
#         frame = back_get_frame(back_path)
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@login_required(login_url='common:login')
@gzip.gzip_page
def detect(request):
    try:
        return StreamingHttpResponse(gen(get_frame), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass
    # while True:
    #     image_data = open(back_path, "rb").read() #"./static/images/image.jpeg"
    #     # ret, img = cv2.VideoCapture(back_path)
    #     # _, jpeg = cv2.imencode('.jpg', img)
    #     # frame = jpeg.tobytes()
    #     # yield(b'--frame\r\n'
    #     #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    #     # response = FileResponse(open('img/data.jpg', 'rb'))
    #     # return StreamingHttpResponse(response, content_type="multipart/x-mixed-replace") # "image/png"

    #     return HttpResponse(image_data, content_type="image/png") #

def nothing_get_frame(cap):
    ret, img = cap.read()
    # img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

@login_required(login_url='common:login')
@gzip.gzip_page
def nothingdetect(request):
    try:
        return StreamingHttpResponse(gen(nothing_get_frame), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass