import torch
import gradio as gr
from torchvision import transforms
from PIL import ImageOps

def load_model():
    model_dict = torch.load('linear_model.pt')
    return model_dict

model = load_model()
convert_tensor = transforms.ToTensor()

def predict(img):
    img =  ImageOps.grayscale(img)
    image_tensor = convert_tensor(img).view(28*28)
    res = image_tensor @ model['weights'] + model['bias']
    res = res.sigmoid()
    return {"It's 3": float(res), "It's 7": float(1-res)}

title = "Is it 7 or 3"
description = '<p><center>Upload an image with a handwritten number: 7 or 3.</center></p>'
examples = ['three.png', 'seven.png']

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             title=title,
             description=description,
             allow_flagging='never',
             examples=examples).launch()