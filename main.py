import os
import cv2
import mediapipe as mp
import time
from deepface import DeepFace


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cam = cv2.VideoCapture(0)
pTime = 0


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)


while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime=cTime
    cv2.putText(img, f'FPS:{int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (52, 52, 52), 3)


    response=DeepFace.analyze(img,actions=("emotion",),enforce_detection=False)
    for face_result in response:
        dominant_emotion = face_result["dominant_emotion"]
        print(f'Dominant Emotion: {dominant_emotion}')

    
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACEMESH_TESSELATION,
            drawSpec,drawSpec)
    

    cv2.imshow("Image", img)
    cv2.waitKey(1)
