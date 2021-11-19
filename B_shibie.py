#石头的识别和分类，用图片与文件夹类别名称进行一一对应训练，模型训练模块
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import Adam
# from keras.preprocessing.imimageimportImageDataGenerator,img_to_array,load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image  import ImageDataGenerator
from keras.models import load_model
from keras.layers import AveragePooling2D
from keras import layers
#新加
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# sess = tf.compat.v1.Session(config=config)


#接着为了提高模型的精准度，需要在图片中进行旋转角度，平移等图形变化形成新的图形拿来训练从而提高精准度代码如下：
train_datagen = ImageDataGenerator(
    rotation_range  = 40,     # 随机旋转度数
    width_shift_range = 0.2, # 随机水平平移
    height_shift_range = 0.2,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 20,       # 随机错切变换
    zoom_range = 0.2,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    fill_mode = 'nearest',   # 填充方式
)

#进行归一化处理
test_datagen = ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
)

#接着定义一些参数变量
IMG_W =244#定义裁剪的图片宽度,记住，输入图片要小
IMG_H = 244 #定义裁剪的图片高度
CLASS = 7 #图片的分类数
EPOCHS = 50#迭代周期###############################################################################
BATCH_SIZE = 6 #批次大小6
TRAIN_PATH = 'D:/Users/YGG/Desktop/pythonProject/class1' #训练集存放路径
TEST_PATH = 'D:/Users/YGG/Desktop/pythonProject/predict' #测试集存放路径
SAVE_PATH = 'D:/Users/YGG/Desktop/pythonProject/rock_selector'#模型保存路径
LEARNING_RATE = 1e-3 #学习率
DROPOUT_RATE = 0 #抗拟合，不工作的神经网络百分比


#接着创建一个神经网络对象
model = Sequential() #创建一个神经网络对象
#接着下面是卷积神经网络的算法部分，首先简单说明下这篇文章所用的卷积神经网络的原理和结构：首先创建第一个卷积层，输入的值是三通道图片的像素矩阵矩阵，即为矩阵的行和列对应图片的宽度和高度，由于图片是三通道图片，故有三个像素矩阵，建立32个卷积核用来和三个像素矩阵相乘，其中卷积核的大小是3*3，即为kernel_size=3，padding=“same”表示对于结果矩阵不丢失边缘数值，激活函数使用relu激活函数，代码即为：
#添加一个卷积层，传入固定宽高三通道的图片，以32种不同的卷积核构建32张特征图，卷积核大小为3*3，构建特征图比例和原图相同，激活函数为relu函数

model.add(Conv2D(input_shape=(IMG_W,IMG_H,3),filters=8,kernel_size=3,padding='same',activation='relu'))

# model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(16, (5, 5), activation='relu'))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dense(84, activation='relu'))
# model.add(Dense(num_class, activation='softmax'))



#再次构建一个卷积层
model.add(Conv2D(filters=8,kernel_size=3,padding='same',activation='relu'))
#构建一个池化层，提取特征，池化层的池化窗口为2*2，步长为2。
model.add(MaxPool2D(pool_size=2,strides=2))
#继续构建卷积层和池化层，区别是卷积核数量为64。
model.add(Conv2D(filters=16,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=16,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))
#继续构建卷积层和池化层，区别是卷积核数量为128。
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten()) #数据扁平化
model.add(Dense(32,activation='relu')) #构建一个具有128个神经元的全连接层
model.add(Dense(16,activation='relu')) #构建一个具有64个神经元的全连接层
model.add(Dropout(DROPOUT_RATE)) #加入dropout，防止过拟合。
model.add(Dense(CLASS,activation='softmax')) #输出层，一共4个神经元，对应4个分类

#接着创建一个ADAM优化器，adam = Adam(lr=LEARNING_RATE)，然后使用交叉熵代价函数，adam优化器优化模型，并提取准确率
adam = Adam(lr=LEARNING_RATE)
#新加的
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])



train_generator = train_datagen.flow_from_directory( #设置训练集迭代器train_datagen.flow_from_directory
    TRAIN_PATH, #训练集存放路径
    # steps_per_epoch=6,   #训练数据的数量除以batch_size，6/1=6
    target_size=(IMG_W,IMG_H), #训练集图片尺寸
    batch_size=BATCH_SIZE #训练集批次
    )

test_generator = test_datagen.flow_from_directory( TEST_PATH,#设置测试集迭代器validation_generator
     #测试集存放路径
    target_size=(IMG_W,IMG_H), #测试集图片尺寸
    batch_size=BATCH_SIZE, #测试集批次
    )


print(train_generator.class_indices) #打印迭代器分类，各分类代号为{#'a黑色煤'0,'b灰黑色泥岩1','c灰色泥质粉砂岩2','d灰色细砂岩3','e浅灰色细砂岩4','f深灰色粉砂质泥岩5','g深灰色泥岩6'}

try:
    model = load_model('{}.h5'.format(SAVE_PATH))  #尝试读取训练好的模型，再次训练,format的内容填入大括号，调用h5文件
    print('model upload,start training!')
except:
    print('not find model,start training') #如果没有训练过的模型，则从头开始训练

model.fit_generator( #模型拟合要不要改model.fit————————————>model.fit_generator
    train_generator,  #训练集迭代器
    steps_per_epoch=len(train_generator), #每个周期需要迭代多少步（图片总量/批次大小=11200/64=175）
        epochs=EPOCHS, #迭代周期
    validation_data=test_generator, #测试集迭代器
    validation_steps=len(test_generator) #测试集迭代多少步
        )
model.save('{}.h5'.format(SAVE_PATH)) #保存模型
print('finish {} epochs!'.format(EPOCHS))

