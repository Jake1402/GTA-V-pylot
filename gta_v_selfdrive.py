import model
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2 as cv
import time 

import cv_screen as cvs
import controls
get_screen = cvs.cv_screen(dimension_resize=(2*54, 2*96))
ctrl = controls.controls()

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

from dataset.datasets_handler import CustDataset
from torch.utils.data import DataLoader
import random

GTA_DS = []
start = 0

'''
This slice of code below belongs in its own class.
It's super effecient. Before this ram usage was at 120GB after
it goes to 10GB meaning it could store all variable on a GPU
with at least 16GB of VRAM.
'''

for files in range(1, 97): #Iterate through all files
    holding = []    #hold all actions after array is fully read
    a = []    #create lists each contatining a frame with its action
    d = []
    w = []
    s = []
    wa = []
    wd = []
    print(f"Reading <{files}>")
    GTA_Train = np.load(f"./dataset/training_data ({files}).npy")    #load file

    for pointer in range(len(GTA_Train)):    #iterate through the array.
        
        if GTA_Train[pointer][1] is None:    #skip over is action is None/undef
            continue

        if np.array(GTA_Train[pointer][1]).argmax() == 0:    #append corresponding actions to list
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

    w = w[:len(wa)][:len(wd)]    #slice the lists down so dataset is balanced.
    a = a[:len(w)]
    d = d[:len(w)]
    s = s[:len(w)]
    wa = wa[:len(w)]
    wd = wd[:len(w)]
    holding = holding + w + s + wa + wd + a + d    #append all lists to holding array.
    del w    #delete all variables
    del a
    del d
    del s
    del wa
    del wd
    for pointer in range(len(holding)):
        #print(str(pointer) + " - " + str(start))
        GTA_Processed = get_screen.processImageToGSFormatNoCanny(holding[pointer][0])
        GTA_Processed = torch.tensor(GTA_Processed).to(torch.float32)
        GTA_DS.append([np.divide(GTA_Processed, 255.0), np.array(holding[pointer][1]).argmax()])    #convert to pytorch varables and append to a numpy list
    del holding    #delete holding

'''^Repeat until all files read^'''


batch_size = 64
dataset = CustDataset(GTA_DS, window_size=1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Number of items in dataset - {len(GTA_DS)}")

del GTA_DS
del GTA_Train

print(f"Is GPU available for use - {torch.cuda.is_available()}")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
agent = model.GRU_Car(device).to(device)
alpha = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(agent.parameters(),lr=alpha, momentum=0.95)
step_lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.75)



def train_epoch(model, dataloader, optimizer, loss_fn, device="cuda"):
    model.train()
    iteration = 0
    for batch, (features, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        iteration += 1

        input = features.to(device)
        out = model.forward(input)
        loss = loss_fn(out, labels.to(device))
        loss.backward()
        optimizer.step()
        
        if batch % 512 == 0:
            print(f"epoch current loss - {loss.sum().item()}, Current Iteration - {iteration}")
        
def test_loop(model, test_dataloader, loss_fn, device="cuda"):

    test_loss = 0
    correct = 0
    size = len(test_dataloader.dataset)

    model.train(False)  #Remove if loading bugs

    with torch.no_grad():
        for X,y in test_dataloader:
            out = model.forward(X.to(device))

            test_loss += loss_fn(out, y.to(device)).item()
            correct += (torch.exp(out).argmax(1) == y.to(device)).type(torch.float).cpu().detach().sum().item()
            break
    test_loss/=size
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>6f} \n")
    return test_loss

path = "path_to_save_models"
epochs = 75
for i in range(epochs):
    initTime = time.time()
    print(f"Training on epoch - {i}")
    train_epoch(model=agent, dataloader=dataloader, optimizer=optimizer, loss_fn=loss_fn, device=device)
    step_lr.step()
    print(f"Training Time took {time.time() - initTime}")
    if i % 5 == 0:
        torch.save(agent.state_dict(), path+f"Model Saving Current Epoch {i}.pt")
        pass

agent.train(False)
print("Deleting")
time.sleep(5)
del dataloader
del dataset

'''
Save models after training so nothing is lost
'''
torch.save(agent.state_dict(), path+f"EPOCH{epochs}, BATCH{batch_size}, LR{alpha}.pt")
input("Training Complete")
time.sleep(5)
counts = 0
'''
run the model in the GTA V enviroment. It will press your keys so ensure you're
in GTA V or it will spam key presses.
'''
while True:
    counts += 1
    X = get_screen.grabScreen()
    X = torch.tensor(X).unsqueeze(dim=0).to(torch.float32).to(device)
    out = agent.forward(X)
    out = torch.exp(out)
    out = out.cpu().detach().numpy()
    print(f"Model Outputs - {out}")
    print(f"Frames Passed - {counts}, Model Argmax - {np.argmax(out)}, Model Keypress - {ctrl.key_dict[np.argmax(out)]}")

    ctrl.control_dict[np.argmax(out)]()

    if counts>300:
        ctrl.NK()
        input("")
        time.sleep(5)
        counts = 0
