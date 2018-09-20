import json
from datetime import datetime
from io import BytesIO

import cv2 as cv
import requests
from PIL import Image


def get_group():
    name2personId = dict()
    name2personId['ChenXiaoXu'] = 'c7caac91-b1d2-42b2-9705-1b98493e4106'
    name2personId['FanBingBing'] = '2177111c-8fef-4011-8ee4-6c8770ae3d84'
    name2personId['HaLiBoTe'] = 'df5ba7ee-6014-4dbf-a3b6-8252334411d4'
    name2personId['HeBen'] = '998c4029-f341-4472-8f5a-7fc7ddfd3535'
    name2personId['LiRuoTong'] = '3814bc25-597f-446b-9d1d-b5e41c14712c'
    name2personId['WuDiJingEr'] = 'bd6cd796-45ef-45b4-9eb0-5f1c9ad73f24'
    name2personId['YuanQuan'] = '96e2f7ad-851c-4400-a0ca-2edaf1f7fa0d'
    name2personId['ZhangBaiZhi'] = 'a99718ce-eb37-48b0-aa5f-248f2bc5b9b9'
    name2personId['ZhangManYu'] = '0a68534b-9ce6-472d-8181-7badb5b64a16'
    name2personId['ZhouDongYu'] = '711e4575-7ac2-41d6-8c4d-e87e11517572'
    name2personId['ZhuYin'] = 'd2313eb2-8004-47c7-a0ea-7b860f0e40d6'

    personId2name = dict()
    for key in name2personId.keys():
        personId2name[name2personId[key]] = key

    return personId2name, name2personId


def draw_boxes(image, faceRectangle, name):
    xmin = faceRectangle['left']
    ymin = faceRectangle['top']
    width = faceRectangle['width']
    height = faceRectangle['height']
    pt1 = (xmin, ymin)
    pt2 = (xmin + width, ymin + height)
    cv.rectangle(image, pt1, pt2, (0, 255, 0), 1)
    cv.putText(image, name, (xmin + 1, ymin + 1), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(image, name, (xmin, ymin), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), lineType=cv.LINE_AA)


def process_one_frame(image):
    detect_url = 'http://localhost:5000/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true'

    image_file = BytesIO()
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    img.save(image_file, "JPEG")
    image_file.seek(0)

    files = [('images', ('test.jpg', image_file, 'image/jpeg'))]
    r = requests.post(detect_url, files=files)
    # print(r.text)
    face_boxes = json.loads(r.text)

    faceId_list = []
    for face in face_boxes:
        faceId_list.append(face['faceId'])

    # print(len(faceId_list))
    # print(faceId_list)

    req = {
        "largePersonGroupId": "6c9ef2d2-bc9c-11e8-ac02-d89ef339b7b0",
        "faceIds": faceId_list,
        "maxNumOfCandidatesReturned": 1,
        "confidenceThreshold": 0.2
    }
    identify_url = 'http://localhost:5000/face/v1.0/identify'
    r = requests.post(identify_url, json=req)
    print(r.text)

    face_identities = json.loads(r.text)
    for face in face_identities:
        if face['candidates']:
            faceId = face['faceId']
            personId = face['candidates'][0]['personId']
            name = personId2name[personId]
            faceRectangle = [f['faceRectangle'] for f in face_boxes if f['faceId'] == faceId][0]
            # print(name)
            # print(faceRectangle)
            draw_boxes(image, faceRectangle, name)

    return image


# if __name__ == '__main__':
#     personId2name, name2personId = get_group()
#     image = cv.imread('images/scene.jpg')
#     image = process_one_frame(image)
#     cv.imwrite('images/face_recognition.png', image)

if __name__ == '__main__':
    personId2name, name2personId = get_group()
    video = 'video/video.mp4'
    cap = cv.VideoCapture(video)
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('video/output.avi', fourcc, 24.0, (368, 640))
    frame_idx = 0
    t1 = datetime.now()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        vis = process_one_frame(frame)
        cv.imshow('frame', vis)
        out.write(vis)
        ch = cv.waitKey(1)
        frame_idx += 1
        if ch == 27:
            break

    t2 = datetime.now()
    delta = t2 - t1
    elapsed = delta.seconds + delta.microseconds / 1E6

    print('fps: ' + str(frame_idx / elapsed))
