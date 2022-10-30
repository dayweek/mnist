import torch
import gradio as gr
from torchvision import transforms
from PIL import ImageOps
import os
from dotenv import load_dotenv

from torch import nn
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # -> 6 channels, 28x28
        self.pool1 = nn.MaxPool2d(2) # -> 6 channels, 14x14
        self.conv2 = nn.Conv2d(6, 16, 5) #-> 16 images, 10x10
        self.pool2 = nn.MaxPool2d(2) # -> 16 channels,  5x5
        self.conv3 = nn.Conv2d(16, 120, 5) #-> 16 images, 1x1
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def __call__(self, x):
        xx = F.relu(self.conv1(x))
        xx = F.relu(self.pool1(xx))
        xx = F.relu(self.conv2(xx))
        xx = F.relu(self.pool2(xx))
        xx = F.relu(self.conv3(xx))
        xx = xx.flatten(1)
        xx = F.relu(self.fc1(xx))
        return self.fc2(xx)

load_dotenv()

hf_writer = gr.HuggingFaceDatasetSaver(os.getenv('HF_TOKEN'), "simple-mnist-flagging")

def load_model():
    model = Lenet()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    return model

model = load_model()
convert_tensor = transforms.ToTensor()

def predict(img):
    img =  ImageOps.grayscale(img).resize((28,28))
    image_tensor = convert_tensor(img).view(1, 1, 28, 28)
    logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1)
    return pred.tolist()[0]

title = "Handwritten digit recognition"
description = '<p><center>Write a single digit in the middle of the canvas</center></p>'

gr.Interface(fn=predict, 
             inputs=gr.Paint(type="pil", invert_colors=True),
             outputs="text",
             title=title,
             flagging_options=["incorrect","ambiguous"],
             flagging_callback=hf_writer,
             description=description,
             allow_flagging='manual').launch()

