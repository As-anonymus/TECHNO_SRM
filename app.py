try:
    import detectron2
except:
    import os 
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')
from matplotlib.pyplot import axis
import requests
import numpy as np
from torch import nn

import requests


import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import streamlit as st
from detectron2.utils.visualizer import ColorMode
import os
import cv2
from PIL import Image, ImageOps
import numpy as np

model_path = "model_final.pth"

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = model_path
st.write("""
         # Car Damage Detection
         """
         )
file = st.file_uploader("Please upload an image file(JPG/PNG/JPEG format)", type=["jpg", "png","jpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)

car_metadata = MetadataCatalog.get("test1")
car_metadata.thing_classes = ['Damage-car','Damage','Others','Undamage']

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE='cpu'
    
predictor = DefaultPredictor(cfg)
def inference(image):
   
    img = np.array(image)
    outputs = predictor(img) 
    v = Visualizer(img[:, :, ::-1],
                   metadata=car_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW  
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)
    st.write("""
              # Output!!
            """
             )
    predictions = inference(image)
    st.image(predictions,use_column_width=True)