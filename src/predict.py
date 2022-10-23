import torch

def load_model():
    model_dict = torch.load('linear_model.pt')
    return model_dict

def predict(image_tensor, model):
    res = image_tensor @ model['weights'] + model['bias']
    return 'seven' if res.sigmoid() > 0.5 else 'three'