import numpy as np
import torch
import torch.nn.functional
import cv_screen as cvs
import cv2 as cv
import math
import PositionalEcoding as pe
import copy

def compileData(int_batch = 0, batch_step = 10, ds_location = "./"):

    path = ds_location
    GTA_DS = []
    height = 82
    width  = 144
    get_screen = cvs.cv_screen(dimension_resize=(height, width))
    #position_encoding = pe.PositionEncoding(width*3, height)

    for files in range(int_batch, batch_step): 
        holding = []
        a = []
        d = []
        w = []
        s = []
        wa = []
        wd = []

        print(path + f"\\training_data-{files}.npy")
        try:
            GTA_Train = np.load(path + f"\\training_data-{int(files)}.npy")
        except Exception:
            print("File Loading Error")
            continue
        print(f"Reading <{files}>")

        for pointer in range(len(GTA_Train)):
            if GTA_Train[pointer][1] is None:
                continue

            if np.array(GTA_Train[pointer][1]).argmax() == 0:
                w.append(GTA_Train[pointer])

            if np.array(GTA_Train[pointer][1]).argmax() == 1:
                s.append(GTA_Train[pointer])

            if np.array(GTA_Train[pointer][1]).argmax() == 2:
                a.append(GTA_Train[pointer])

            if np.array(GTA_Train[pointer][1]).argmax() == 3:
                d.append(GTA_Train[pointer])

            if np.array(GTA_Train[pointer][1]).argmax() == 4:
                wa.append(GTA_Train[pointer])

            if np.array(GTA_Train[pointer][1]).argmax() == 5:
                wd.append(GTA_Train[pointer])

        w = w[:math.floor(len(wa)*1.2)][:math.floor(len(wd)*1.2)]
        a = a[:len(w)]
        d = d[:len(w)]
        wa = wa[:len(w)]
        wd = wd[:len(w)]
        holding = holding + w + wa + wd + a + d + s + s + s
        del w
        del a
        del d
        del s
        del wa
        del wd
        for pointer in range(len(holding)):
            GTA_Processed = get_screen.processImageChannelLast(holding[pointer][0])
            GTA_Processed = np.array(GTA_Processed, dtype=np.float32)
            GTA_Processed = GTA_Processed.reshape(3, 82, 144)
            GTA_DS.append([np.divide(GTA_Processed, 255.0), np.array(holding[pointer][1]).argmax()])
        del holding


    t_GTA_DS = copy.deepcopy(GTA_DS)
    '''
    zeros = np.zeros((height, width, 3))
    for i in range(len(t_GTA_DS)):

        in_holder = t_GTA_DS[i][0]

        if i < 12:
            in_holder = copy.deepcopy(
                np.hstack(
                    [
                        in_holder, 
                        zeros, 
                        zeros
                    ]
                )
            )
        else:
            in_holder = copy.deepcopy(
                np.hstack(
                    [
                        in_holder, 
                        np.array(GTA_DS[::][::][i-5][0], dtype=np.float32), 
                        np.array(GTA_DS[::][::][i-10][0], dtype=np.float32)
                    ]
                )
            )
        #in_holder = position_encoding.forward3D(torch.tensor(in_holder))   """Enable if you'd like to do attention"""
        in_holder = in_holder.reshape(3, 82, 432)
        t_GTA_DS[i][0] = in_holder
    '''
    GTA_DS = t_GTA_DS
    del t_GTA_DS

    batch_file = []
    print(len(GTA_DS))
    for iter in range(int_batch, batch_step):
        print(f"Saving File, currently on file {iter} of {batch_step}")
        batch_file = np.asarray(GTA_DS[(iter-int_batch)*math.floor(len(GTA_DS)/(batch_step-int_batch)):(iter+1-int_batch)*math.floor(len(GTA_DS)/(batch_step-int_batch))], dtype="object")
        print(f"Saving at - ./dataset/training_data ({int(iter)}).npy")
        print(f"Shape of saved data - {batch_file.shape}")
        np.save(f"./dataset/training_data ({int(iter)}).npy", batch_file, allow_pickle=True)


def displayProcessedFrames(frames, dims=(82, 144)):
    frame_delay = 1 / 240

    for img in frames:
        if isinstance(img, str):
            frame = cv.imread(img[0])
            if frame is None:
                print(f"Error: Unable to read image at {img}")
                continue
        else:
            frame = img[0]

        frame = cv.cvtColor(np.array(frame*200, dtype=np.uint8), cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (dims[1], dims[0]))
        cv.imshow("GTA V Processed Frames", frame)

        if cv.waitKey(int(frame_delay * 1000)) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    print("Executing")
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    for i in range(20):
        print(f"Starting Batch = {10*i}, batch step = {10*(i+1)}")
        compileData(int_batch=10*i, batch_step=10*(i+1), ds_location="D:\\GTA_V_DS\\self-driving-GTA-V\\Training Data(1-100)")