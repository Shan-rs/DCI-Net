from MyModel import DynamicDecoder
from PIL import Image
from config import Config
import glob
import torch
import torchvision
import os
import utils.save_load as sl
import numpy as np
from torchvision import transforms as T
import time
import torch.nn.functional as F
import math
from thop import profile

start = time.time()
imgs = glob.glob('')
#def dehaze(imgs):
opt = Config()
transform = T.Compose([T.ToTensor()])
model = DynamicDecoder(36)
model, _, _, _ = sl.load_state(opt.load_model_path, model)

if torch.cuda.is_available():
    model = model.cuda()
#print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
output_path = ''
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
for img in imgs:
    hazy = Image.open(img)
    output_name = img.split('/')[-1]
    hazy = transform(hazy).unsqueeze(0)
    
    if torch.cuda.is_available():
        hazy = hazy.cuda()

    with torch.no_grad():
        out1, out2, out3, out_image = model(hazy)
        torchvision.utils.save_image(out_image, output_path + '/' + output_name)

end = time.time()
print(str(end-start))
print("Done!")


