# GTA V Pylot
GTA V Pylot is a convolutional model who's only objective is to drive a car in the GTA V world. To do this I used the [self-driving-GTA-V](https://huggingface.co/datasets/sartajbhuvaji/self-driving-GTA-V) by [Sartaj](https://github.com/SartajBhuvaji). An important limitation of this car is that a lot of training data had to be removed in order to balance the dataset. I plan to correct this issue in future iteration by changing how I decide to balance the dataset. Another important limitation is the lack of memory the network has, the network does not feature any frame stacking and as a result only acts on a single image at a time. The network is a feed forward with no recurrent connections.

## How to install
To install the model you need to have all the requirements. The requirements for this model are all contained in [requirements](./requirements.txt). I used a Anaconda environment for my install so I'd recommend you do the same or do something similar. To begin install [PyTorch](https://pytorch.org/get-started/locally/) on your machine, it's important you use at least PyTorch 2 or higher. To do this follow the guide found on their website.
Once PyTorch has been installed simply follow the command below.
```
pip install -r requirements.txt
```
This will begin to install all the necessary libraries in order to successfully run GTA-V-Pylot. I've also included a simple model that has marginal success on some areas of the map. Specifically the coastal highway on the west side of the map. The model seems to do alright on highways but can struggle when driving in the city. 

## Running the model
To run the model simply run [load_and_play.py](load_and_play.py) if you want to load and run your own model or [load_example.py](./model/load_example.py) if you'd like to run the pretrained example model.

## Training your own model

#### Preparing Data
If you'd like to train your own model, I'd suggest you download the above mentioned dataset in the order it's displayed in on the hugging face repo. Then rename your files to           "training_data (N).npy" where N is the file number. I found this to be an effective way to to rename my training data and maintain the order of the data. This step is crucial if you'd like to do frame stacking or work with recurrent model in the future. To batch rename the data do "Ctrl+A" and right click the first file and rename to "training_data.npy" windows will automatically reformat the files to above mentioned file name. Finally save the renamed data to the [dataset](./dataset) folder.
The data will be automatically processed, balanced and edited when you run [gta_v_selfdrive.py](gta_v_selfdrive.py)
file. The algorithm used for this was heavily inspired by [Sentdex](https://github.com/sentdex) data balancing for his GTA V self driving car [Charles](https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a). 

A simple overview of this algorithm can be expressed with just a few bullet points.
 - Load the first numpy file into memory. This numpy file contains an array that holds a few thousand image and action pairs.
 - Create 7 lists. One for holding the result after the lists have been process, and 6 more each one representing a possible action the model can make.
 - Iterate over everything in the first array and sort it into the 6 respective lists.
 - Slice the lists down so they're all the same size. (Data is lost here making RNNs or frame stacking difficult)
 - Merge all the lists together and store the result into holding list.
 - Normalise the images and convert the action array into a singular argmax value.
 - Finally return the holding list.
### Training your model
I'd heavily suggest modifying the [model.py](model.py) file and changing the models architecture to see what kind of results you may get. I tried quite a few combinations and some did perform exceptionally well managing to drive for quite some time without crashing or losing control. But to train the model run [gta_v_selfdrive.py](gta_v_selfdrive.py) this file handles data processing and training, the hyperparameters used for all my models are,
```
alpha - 0.001
loss function - nn.CrossEntropyLoss()
optimizer - Stochastic Gradient Descent with momentum 0.95
LR Scheduler - stepped every 20 epochs with a drop of 0.75
```
I found these values worked quite well, unfortunately I noticed that Adam could be unstable but I did get some good results with it. Perhaps and most likely it was my fault for improperly picking the write hyper parameters. Training for me took between 1.5 to 6 hours depending on what model I decided to create and how much data I wanted to use. My system specs are,
```
CPU - AMD Ryzen 5 5600X
GPU - NVidia RTX 3090 24GB
RAM - 32GB DDR4 3200MHz
STORAGE - ADATA SX8200 
```

# Final Remarks and References
I'd heavily suggest you use a single player mod menu or trainer when using this in single player. It allows for much more ease of use when choosing car or disabling traffic to see the models full potential. Below I've left some good references to look at and sources that heavily inspired this project.
[Sentdex Balancing Data](https://www.youtube.com/watch?v=wIxUp-37jVY)
[GTA V Selfdrive Dataset](https://huggingface.co/datasets/sartajbhuvaji/self-driving-GTA-V)
[Sentdex - Python Plays GTA V](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)
[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
