import cv2 as cv
import numpy as np
import torch
from PIL import ImageGrab
import pyautogui

class cv_screen:
    def __init__(self, positions=(0,30), image_dimension=(800,600), dimension_resize=(54, 96), channels=1) -> None:
        self.positionX = positions[0]
        self.positionY = positions[1]
        self.imageWidth = image_dimension[0]
        self.imageHeight = image_dimension[1]
        self.resizeWidth = dimension_resize[0]
        self.resizeHeight = dimension_resize[1]
        self.channels = channels
                                        
        self.ROI_vertices = np.array([[0,1], [0,0.7], [0.3,float(4/9)], [0.7,float(4/9)], [1,0.7], [1,1]])
        self.ROI_dim = np.array([dimension_resize[1], dimension_resize[0]])

    def grabScreen(self) -> np.array:
        screenArray = np.array(ImageGrab.grab(bbox=(self.positionX, self.positionY, self.imageWidth, self.imageHeight)))
        screenArray = cv.resize(screenArray, (self.resizeHeight, self.resizeWidth))

        cannyChannel = np.reshape(self.ReturnCanny(screenArray), (1, self.resizeWidth, self.resizeHeight))
        screenArray = cv.cvtColor(screenArray, cv.COLOR_BGR2GRAY)
        screenArray = np.reshape(screenArray, (1, self.resizeWidth, self.resizeHeight))
        screenArray = np.concatenate((screenArray, cannyChannel), axis=0)
        return screenArray
    


    def processImageToGSFormat(self, image) -> np.array:
        screenArray = cv.resize(image, (self.resizeHeight, self.resizeWidth))

        cannyChannel = np.reshape(self.ReturnCanny(screenArray), (1, self.resizeWidth, self.resizeHeight))
        screenArray = cv.cvtColor(screenArray, cv.COLOR_BGR2GRAY)
        screenArray = np.reshape(screenArray, (1, self.resizeWidth, self.resizeHeight))
        screenArray = np.concatenate((screenArray, cannyChannel), axis=0)
        return screenArray
    
    def processImageToGSFormatNoCanny(self, image) -> np.array:
        screenArray = cv.resize(image, (self.resizeHeight, self.resizeWidth))
        screenArray = cv.cvtColor(screenArray, cv.COLOR_BGR2GRAY)
        screenArray = np.reshape(screenArray, (1, self.resizeWidth, self.resizeHeight))
        return screenArray
    
    def grabScreenNoCanny(self):
        screenArray = np.array(ImageGrab.grab(bbox=(self.positionX, self.positionY, self.imageWidth, self.imageHeight)))
        screenArray = cv.resize(screenArray, (self.resizeHeight, self.resizeWidth))
        screenArray = cv.cvtColor(screenArray, cv.COLOR_BGR2GRAY)
        screenArray = np.reshape(screenArray, (1, self.resizeWidth, self.resizeHeight))
        return screenArray

    #Region Of Interest - Sentdexs example
    def ROI(self, frame, vertices):
        mask = np.zeros_like(frame)
        cv.fillPoly(mask,vertices, 255)
        masked = cv.bitwise_and(frame, mask)
        return masked

    def draw_lines(self, frame, lines):
        try:
            for line in lines:
                coords = line[0]
                cv.line(frame, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
        except:
            pass

    def ReturnCanny(self, currentFrame) -> np.array:
        currentFrame = cv.GaussianBlur(currentFrame, (3,3), 0)
        currentFrame = cv.Canny(currentFrame, 190, 255, L2gradient=True)
        vertice_scale = np.int32(self.ROI_vertices*self.ROI_dim)
        currentFrame = self.ROI(currentFrame, [vertice_scale])

        currentFrame = cv.GaussianBlur(currentFrame, (3,3), 0)
        #lines = cv.HoughLinesP(currentFrame, 1, np.pi/180, 180, None, 20, 20)
        #self.draw_lines(currentFrame, lines)

        return currentFrame

if __name__ == "__main__":
    while True:
        screenGrab = cv_screen(dimension_resize=(2*54, 2*96))
        frame = screenGrab.grabScreenNoCanny()
        print(np.shape(frame))
        #frameComb = np.hstack((
         #   cv.cvtColor(frame[0], cv.COLOR_GRAY2RGB),
          #  cv.cvtColor(frame[1], cv.COLOR_GRAY2RGB)
        #))
        cv.imshow("GTA V Self Driving Car", frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            cv.destroyAllWindows()
            break