import model
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2 as cv

import cv_screen as cvs
import controls
import time

get_screen = cvs.cv_screen(dimension_resize=(2*54, 2*96))
ctrl = controls.controls()

print(f"Is GPU available for use - {torch.cuda.is_available()}")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

agent = model.GRU_Car(device=device).to(device=device)
agent.load_state_dict(torch.load("example_model.pt", weights_only=False))
agent.eval()
input("Model Loaded ~ Press Enter to Continue")

time.sleep(5)
counts = 0
with torch.no_grad():
    while True:
        time.sleep(0.001)
        ctrl.NK()
        counts += 1
        X = get_screen.grabScreenNoCanny()
        in_ = X
        X = torch.tensor(X).unsqueeze(dim=0).to(torch.float32).to(device)
        out = agent.forward(X)
        out = torch.exp(out)
        out = out.cpu().detach().numpy()
        print(f"Model Outputs - {out}")
        print(f"Frames Passed - {counts}, Model Argmax - {np.argmax(out)}, Model Keypress - {ctrl.key_dict[np.argmax(out)]}")

        ctrl.control_dict[np.argmax(out)]()

        if cv.waitKey(1) & 0xff == ord('q'):
            cv.destroyAllWindows()
            break

        if counts>400:
            ctrl.NK()
            input("")
            time.sleep(5)
            counts = 0