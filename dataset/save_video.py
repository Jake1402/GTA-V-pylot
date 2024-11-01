import cv2 as cv
import numpy as np

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


GTA_DS = []
for files in range(1, 30):
    GTA_Train = np.load(f"C:\\Users\jakey\\Desktop\\Programming stuff\\GTAV SelfDriving\\dataset\\training_data ({files}).npy")
    for pointer in range(len(GTA_Train)):
        if GTA_Train[pointer][1] is None:
            continue
        #GTA_Processed = cv.resize(GTA_Train[pointer][0], (get_screen.resizeWidth, get_screen.resizeHeight), interpolation=cv.INTER_LINEAR)
        #GTA_Processed = torch.tensor(np.reshape(GTA_Processed, (3, get_screen.resizeHeight, get_screen.resizeWidth))).to(torch.float32)
        GTA_Processed = (GTA_Train[pointer][0])
        #print(np.shape(GTA_Processed))
        GTA_DS.append(GTA_Processed)


out = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (240, 135))
for frame in GTA_DS:
    frame = np.array(frame)
    frame = cv.resize(frame, (240, 135))
    out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
out.release()