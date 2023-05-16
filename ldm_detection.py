# import libraries
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from os import listdir

# load CNN model
from tensorflow.keras.models import load_model
cnn = load_model('ldm_2008c3_Sigmoid.h5')

# initialisation
stat_simple = np.zeros((50,10), dtype=int)
stat_complex = np.zeros((50,10), dtype=int)
stat_total = np.zeros((50,10), dtype=int)
images=listdir('2007c1/all/')
scount=0
ccount=0
y_pred=[i for i in range(4216)]
y_true=[i for i in range(4216)]
k=0

# apply prediction classification
for img in images:
    orb=int(img[10:12])-1 # orbit number
    path_image='2007c1/all/'+ img
    test_image = load_img(path_image, target_size = (64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image,verbose = 0)
    if result[0][0] > 0.7:
        # prediction is simple
        prediction=' is simple'
        s=int(img[12:14])
        scount=scount+1
        stat_simple[[orb],s-1]+=1
    else:
        # prediction is complex
        prediction=' is complex'
        c=int(img[12:14])
        ccount=ccount+1
        stat_complex[[orb],c-1]+=1
    # print(img, prediction )

stat_total=stat_simple+stat_complex

#=====================================================
# model performance with confusion matrix
#import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt


y_pred=[i for i in range(4216)] # 4216 (2007) si 5194 (2008)
y_true=[i for i in range(4216)]
images=listdir('2007c1/all/')
k=0

# apply prediction classification for y_pred
for img in images:
    orb=int(img[10:12])-1 # orbit number
    path_image='2007c1/all/'+ img
    test_image = load_img(path_image, target_size = (64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image,verbose = 0)
    if result[0][0] > 0.7:
        # prediction is simple
        prediction='s'
    else:
        # prediction is complex
        prediction='c'
  
    #predict y_pred
    y_pred[k]=prediction
    # extract y_true
    img_name1=img.split('_')[-1]
    img_name2=img_name1.split('.')[-2 ]
    y_true[k]=img_name2
    k+=1

cm = confusion_matrix(y_true, y_pred)
cm=np.flip(confusion_matrix(y_true, y_pred))
print(cm)
print(matthews_corrcoef(y_true,y_pred))

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(cm, display_labels="LDM")
cmp.plot(ax=ax)