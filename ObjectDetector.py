# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import os
import numpy as np
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import urllib

parser = argparse.ArgumentParser(description="Bounding Box detection")

parser.add_argument('--image_dir', action = "store", dest = "directory", default = ".", help="Input image directory")

parser.add_argument('--det', action = "store", dest = "result", default = ".", help = "Store the result in this directory") 

args = parser.parse_args()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# ## Load RetinaNet model

# In[2]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())


# ## Run detection on example

# In[4]:


dct = args.directory
import os
flist = os.listdir(dct)
x1 = []
x2 = []
y1 = []
y2 = []
for f in flist:
    dest = dct + '/' + f
    temp = read_image_bgr(dest)

    # preprocess image for network
    temp = preprocess_image(temp)
    temp, scale = resize_image(temp)
    #start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(temp, axis=0))
    #print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale


    max_score = scores[0][0]
    box = boxes[0][0].astype(int)
    csv_box = np.zeros(shape=(4,), dtype = int)
    x1.append(box[0])
    x2.append(box[2])
    y1.append(box[1])
    y2.append(box[2])
#     print(box)
#     print(csv_box)
#     label = labels[0][0]

#     color = label_color(label)

#     b = box.astype(int)
#     draw_box(draw, b, color=color)

#     caption = "{:.3f}".format(max_score)
#     draw_caption(draw, b, caption)

#     plt.figure(figsize=(15, 15))
#     plt.axis('off')
#     plt.imshow(draw)
#     plt.show()

    


# In[5]:


import pandas as pd


# In[6]:


d = {'image_name': flist, 'x1': x1, 'x2':x2, 'y1':y1, 'y2':y2}


# In[7]:


df = pd.DataFrame(data=d)


# In[8]:
res_dir = args.result
path_to_write = res_dir + '/test.csv'

df.to_csv(path_to_write, index = False)

