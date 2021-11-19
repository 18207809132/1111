#岩石类别预测模块
from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from keras.preprocessing import image
SAVE_PATH = 'D:/Users/YGG/Desktop/pythonProject/rock_selector'#模型保存路径
model = load_model('{}.h5'.format(SAVE_PATH))

label = np.array(['黑色煤','灰黑色泥岩','灰色泥质粉砂岩','灰色细砂岩','浅灰色细砂岩','深灰色粉砂质泥岩','深灰色泥岩'])
#'a黑色煤','b灰黑色泥岩','c灰色泥质粉砂岩','d灰色细砂岩','e浅灰色细砂岩','f深灰色粉砂质泥岩','g深灰色泥岩'
def image_change(image):#定义函数将图片转换成可识别的矩阵
 image = image.resize((244, 244))
 image = img_to_array(image)
 image = image / 255
 image = np.expand_dims(image, 0)
 return image

for pic in os.listdir('./predict/'):

    print('图片真实结果为',pic)
    image = load_img('./predict/'+pic)
    plt.imshow(image)
    image = image_change(image)
    myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
    plt.title(label[model.predict_classes(image)], fontproperties=myfont)
    plt.show()
    print('预测结果为',label[model.predict_classes(image)])
    print('----------------------------------')
#10个样本模型精度曲线图
x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y=[1,0,0,0,1,0,1,0,1,1]
plt.plot(x,y,label='误差')
plt.show()#画图

