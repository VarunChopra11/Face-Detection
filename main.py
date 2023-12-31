import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACEMESH_TESSELATION,
            drawSpec,drawSpec)
            # Code Landmarks using this:
            '''for id,lm in enumerate(facelms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                if id in [109,338,477,1,5,10]:            #Add landmarks in this list to print their id and position.
                    print(id,x,y)'''                      #This will print landmark's id,horizontal position,vertical position 

    cv2.imshow("Image", img)
    cv2.waitKey(1)
