import cv2
import numpy as np
from objloader_simple import *
import os

MIN_MATCHES = 50
SCALE = 75
WIDTH = 640
HEIGHT = 480

def getProjectionMatrix(camPar, hom):
    rt = np.dot(np.linalg.pinv(camPar), hom * (-1))
    lx = np.sqrt(np.linalg.norm(rt[:, 0], 2) * np.linalg.norm(rt[:, 1], 2))
    rt1, rt2, tr = rt[:, 0] / lx, rt[:, 1] / lx, rt[:, 2] / lx
    rt1 = np.dot((rt1 + rt2) / np.linalg.norm((rt1 + rt2), 2) +
        np.cross((rt1 + rt2), np.cross(rt1, rt2)) / np.linalg.norm(np.cross((rt1 + rt2),\
        np.cross(rt1, rt2)), 2), 1 / np.sqrt(2))
    rt2 = np.dot((rt1 + rt2) / np.linalg.norm((rt1 + rt2), 2) -\
        np.cross((rt1 + rt2), np.cross(rt1, rt2)) / np.linalg.norm(np.cross((rt1 + rt2),\
        np.cross(rt1, rt2)), 2), 1 / np.sqrt(2))
    rt3 = np.cross(rt1, rt2)
    return np.dot(camPar, np.stack((rt1, rt2, rt3, tr)).T)

def getPoints(img, obj, proj, mod):
    vert = obj.vertices
    scale = np.eye(3) * SCALE
    height, width = mod.shape
    points = []
    for face in obj.faces:
        pts = np.array([vert[vertex - 1] for vertex in face[0]])
        pts = np.dot(pts, scale) + np.array([(width / 2), (height / 2), 0])
        dst = cv2.perspectiveTransform(pts.reshape(-1, 1, 3), proj)
        dst[:, :, 0] = np.clip(dst[:, :, 0], 0.0, WIDTH)
        dst[:, :, 1] = np.clip(dst[:, :, 1], 0.0, HEIGHT)
        dst = np.int32(dst)
        points.append(dst)
    return points

def averagePoint(points):
    avgPoints = []
    for pt in points:
        avgPoints.append((np.array(pt[0][0]) + np.array(pt[1][0]) + np.array(pt[2][0])) // 3)
    avg = avgPoints[0]
    for i in range(1, len(avgPoints)):
        avg += avgPoints[i]
    return (avg / len(avgPoints))

if __name__ == '__main__':
    fl = np.load('./data/calib.npz')
    print(fl['mtx'])
    # print(fl['mtx'])
    # camPar = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    camPar  = fl['mtx']
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    model = cv2.imread('hiro.png', cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(model, None)
    end = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)
    kpEnd, desEnd = orb.detectAndCompute(end, None)
    
    obj = OBJ('car.obj', swapyz=True)
    points = []
    avgEnd = None
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output3.avi', fourcc,30,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


    while True:
        ret, frame = cap.read()
        kpFrame, desFrame = orb.detectAndCompute(frame, None)
        
        matchesEnd = bf.match(desEnd, desFrame)
        matchesEnd = sorted(matchesEnd, key=lambda x: x.distance)

        if len(matchesEnd) > MIN_MATCHES:
            src = np.float32([kpEnd[m.queryIdx].pt for m in matchesEnd]).reshape(-1, 1, 2)
            dst = np.float32([kpFrame[m.trainIdx].pt for m in matchesEnd]).reshape(-1, 1, 2)
            relevant = []
            tempAvg = np.array([0.0, 0.0])
            for i in range(10):
                relevant.append(dst[i][0])
                tempAvg += dst[i][0]
            tempAvg = tempAvg / 10
            if avgEnd is None:
                avgEnd = np.int16(tempAvg)
            else:
                avgEnd = np.int16((avgEnd + tempAvg) / 2)
            cv2.circle(frame, tuple(avgEnd), 5, (0, 0, 255), -1)
        
        matches = bf.match(des, desFrame)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCHES:
            src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kpFrame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            hom = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)[0]
            
            if hom is not None:
                projection = getProjectionMatrix(camPar, hom)
                newPts = getPoints(frame, obj, projection, model)
                if points == []:
                    points = newPts
                else:
                    for i, pt in enumerate(points):
                        points[i] = (pt + newPts[i]) // 2
                for pt in points:
                    cv2.fillConvexPoly(frame, pt, (127, 127, 63))
                cv2.circle(frame, tuple(np.int16(averagePoint(points))), 5, (128, 190, 50), -1)
                if avgEnd is not None:
                    motion = avgEnd - np.int16(averagePoint(points))
                    for pt in points:
                        move0 = np.int16(motion[0] * 1.2)
                        # print("here : ",pt[:,:,0], move0, pt[:,:,0]+move0)
                        pt[:, :, 0] += move0
                        pt[:, :, 1] += np.int16(motion[1] * 1.2)
        
        
        # capp = cv2.VideoCapture(0)
       

        window_name = 'projectAR'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()