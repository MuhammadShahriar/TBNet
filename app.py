import shutil
import uvicorn 
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates

from keras.models import load_model
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import RMSprop

app = FastAPI() 

templates = Jinja2Templates(directory="templates") 

app.mount("/static", StaticFiles(directory="static"), name="static") 



class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tensorflow.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads



opt = GCRMSprop(learning_rate=1e-4) 
model1 = load_model(os.path.join("./Classification-Models/", "model_1.h5"), custom_objects={'GCRMSprop': opt})
model1.load_weights(os.path.join("./Classification-Models/", "model_1.h5"))


model2 = load_model(os.path.join("./Classification-Models/", "model_2.h5"), custom_objects={'GCRMSprop': opt})
model2.load_weights(os.path.join("./Classification-Models/", "model_2.h5"))


model3 = load_model(os.path.join("./Classification-Models/", "model_3.h5"), custom_objects={'GCRMSprop': opt})
model3.load_weights(os.path.join("./Classification-Models/", "model_3.h5"))


model4 = load_model(os.path.join("./Classification-Models/", "model_4.h5"), custom_objects={'GCRMSprop': opt})
model4.load_weights(os.path.join("./Classification-Models/", "model_4.h5"))

model5 = load_model(os.path.join("./Classification-Models/", "model_5.h5"), custom_objects={'GCRMSprop': opt})
model5.load_weights(os.path.join("./Classification-Models/", "model_5.h5"))

def PrepareImage( resize_shape = tuple(), color_mode = "rgb", image_path = '.static/Lung.png'):
    img = None

    resized_image = cv2.resize(cv2.imread(image_path),(260,260))
    # print(resized_image)
    # resized_image = resized_image/255.
    if color_mode == "gray":
        img = resized_image[:,:,0]
    elif color_mode == "rgb":
        img = resized_image[:,:,:]
    
    print(img.shape)
    print(resized_image.shape)
    return resized_image


def Classification():
    image_path = os.path.join("./static", "Lung.png")
    img = PrepareImage(resize_shape = (260*260, 3), color_mode = "gray", image_path = image_path)

    predictionNormal = 0
    predictionTuberculosis = 0


    Y_pred = model1.predict(img.reshape(1, 260, 260, 3))
    y_pred = np.argmax(Y_pred, axis=1)

    if ( y_pred[0] == 0 ) :
        predictionNormal += 1
    else :
        predictionTuberculosis += 1
    
    Y_pred = model2.predict(img.reshape(1, 260, 260, 3))
    y_pred = np.argmax(Y_pred, axis=1)

    if ( y_pred[0] == 0 ) :
        predictionNormal += 1
    else :
        predictionTuberculosis += 1

    
    Y_pred = model3.predict(img.reshape(1, 260, 260, 3))
    y_pred = np.argmax(Y_pred, axis=1)

    if ( y_pred[0] == 0 ) :
        predictionNormal += 1
    else :
        predictionTuberculosis += 1


    
    Y_pred = model4.predict(img.reshape(1, 260, 260, 3))
    y_pred = np.argmax(Y_pred, axis=1)

    if ( y_pred[0] == 0 ) :
        predictionNormal += 1
    else :
        predictionTuberculosis += 1

    
    
    Y_pred = model5.predict(img.reshape(1, 260, 260, 3))
    y_pred = np.argmax(Y_pred, axis=1)

    if ( y_pred[0] == 0 ) :
        predictionNormal += 1
    else :
        predictionTuberculosis += 1

   

    print(predictionNormal)
    print(predictionTuberculosis)

    if ( predictionTuberculosis > predictionNormal ) :
        return "Tuberculosis"


    return "Normal"


def Segmantation(model, img_array, img_num, img_side_size = 256):
    
    pred = model.predict(img_array.reshape(1,img_side_size,img_side_size,1))
    pred[pred>0.5] = 1.0
    pred[pred<0.5] = 0.0
    img1 = pred.reshape(img_side_size, img_side_size)
    
    pil_image=Image.fromarray(img1*255)
    pil_image = pil_image.convert("L")
    pil_image.save("./static/Mask.png", "PNG")


    img_org = cv2.imread("./static/Image.png")
    img_mask = cv2.imread("./static/Mask.png")
    
    img_org = cv2.resize(img_org, (256, 256), interpolation = cv2.INTER_AREA)
    img_mask = cv2.resize(img_mask, (256,256), interpolation = cv2.INTER_AREA)

    lung = cv2.bitwise_and(img_mask, img_org)
    pil_image=Image.fromarray(lung)
    pil_image.save("./static/Lung.png")
    
    return pred


def CreateMask( resize_shape = tuple(), color_mode = "rgb", image_path = "./Image.png"):
    img = None

    resized_image = cv2.resize(cv2.imread(image_path),resize_shape)
    resized_image = resized_image/255.
    if color_mode == "gray":
        img = resized_image[:,:,0]
    elif color_mode == "rgb":
        img = resized_image[:,:,:]
    

    return img

def SegmentImage():
    model = load_model(os.path.join("./U-net-Model/", "model.h5"))
    model.load_weights(os.path.join("./U-net-Model/", "weight.h5"))
    img = CreateMask(resize_shape = (256,256), color_mode = "gray", image_path = os.path.join("./static", "Image.png"))
    
    IMG_NUM = 2 #Melhor img_num 12 (0.98) Pior img_num 10 (0.9)
    Segmantation(model, img_array = img, img_num = IMG_NUM, img_side_size = 256)


@app.get('/tbnet', response_class=HTMLResponse) 
def get_awsome_form(request: Request): 
    data = {
        "result": "Result will be shown here",
        "mask": "",
        "lung": "",
        "image": ""
    }
    return templates.TemplateResponse("index.html", {"request": request, "result" : data}) 



@app.post('/tbnet', response_class = HTMLResponse) 
def post_awsome_form(request: Request, file : UploadFile = File(...)):
    with open(f'./static/Image.png', 'wb') as out_file:
        shutil.copyfileobj(file.file, out_file)

    SegmentImage()
    result = Classification()

    data = {
        "result": result,
        "mask": "Mask.png",
        "lung": "Lung.png",
        "image": "Image.png"
    }

    return templates.TemplateResponse("index.html", {"request": request, "result" : data}) 


if __name__ == '__main__':
    uvicorn.run(app) 
