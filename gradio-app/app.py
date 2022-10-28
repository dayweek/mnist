import torch
import gradio as gr
from torchvision import transforms
from PIL import ImageOps
import os
from dotenv import load_dotenv

load_dotenv()

hf_writer = gr.HuggingFaceDatasetSaver(os.getenv('HF_TOKEN'), "simple-mnist-flagging")

def load_model():
    model_dict = torch.load('linear_model.pt')
    return model_dict

model = load_model()
convert_tensor = transforms.ToTensor()

def predict(img):
    img =  ImageOps.grayscale(img).resize((28,28))
    image_tensor = convert_tensor(img).view(28*28)
    res = image_tensor @ model['weights'] + model['bias']
    res = res.sigmoid()
    return {"It's 3": float(res), "It's 7": float(1-res)}

title = "Is it 7 or 3"
description = '<p><center>Write a number, 7 or 3, in the middle.</center></p>'

gr.Interface(fn=predict, 
             inputs=gr.Paint(type="pil", invert_colors=True),
             outputs=gr.Label(num_top_classes=2),
             title=title,
             flagging_options=["incorrect","ambiguous"],
             flagging_callback=hf_writer,
             description=description,
             allow_flagging='manual').launch()