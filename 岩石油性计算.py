#计算油性占比的模型
import glob
import matplotlib.pyplot as plt
import cv2
import os
import eventlet
# celery -A <mymodule> worker -l info -P eventlet
def calculate(photo, digit=6):
    """"读取图片并完成含油性占比等一系列的计算"""
    # 读取图片
    im = cv2.imread(photo)

    # 计算黑白像素与总像素个数，由于用R G B 3个通道处理，所以除以3
    black = len(im[im <= 127]) / 3
    white = len(im[im > 127]) / 3
    print('不含油性面积（像素个数）：', round(black), '\t\t\t含油性面积（像素个数）：', round(white))  # 打印出黑白像素个数,并用3个空白间隔开
    print('岩石总面积（像素个数）：', round(black) + round(white))

    # 计算黑白像素占比，黑：不含油性面积，白：含油性面积
    all_pixel = black + white
    rate1 = black / all_pixel
    rate2 = white / all_pixel
    fvc = round(rate2 * 100, digit)

    # 打印黑色与白色像素所占比例，并保留6位小数
    print('非油性占比:', round(rate1 * 100, digit), '%')
    # print('油性占比:', round(rate2 * 100, digit), '%')
    # global y
    # y=list()
    # y = fvc
    # 插入1个空行并打印岩石油性占比
    print('\n油性占比=', fvc, '%')

folders = glob.glob('./you_xing')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.bmp'):
        imagenames_list.append(f)

read_images = []
for image in imagenames_list:
    read_images.append(cv2.imread(image,1))

for j in range(10):#选取十张含油性的荧光图片进行识别预测含油量
    plt.imshow(read_images[j])
    # plt.show()
# 导入原始图片
# im_1 = cv2.imread('./predict/37-1.bmp ')  # imread()函数有两个参数，第一个参数是图片路径，第二个参数表示读取图片的形式

    # 提取图像的三个通道
    B, G, R = cv2.split(read_images[j])  # split()函数参数为要进行分离的图像矩阵 ; opencv中，RGB三个通道是反过来的

    # 计算油性指数
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    gray = cive.astype('uint8')  # astype()函数用于转换数据类型

    # 阈值分割，将阴影像素点变为0，油性像素点为1;ret为白，th为黑
    ret, th = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU, None)  # 将灰度图gray中灰度值小于0的置0，大于0的置1，第4个实参为阈值类型

    # 保存分离的结果图
    b = B * th
    g = G * th
    r = R * th
    img_1 = cv2.merge([b, g, r])
    cv2.imwrite('result1.jpg', img_1)

    # 图片二值化
    img = cv2.imread('result1.jpg', 0)
    plt.imshow(img)
    plt.show()

    # 使用大津阈值分割
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('result2.jpg', th2)

    # 使用5x5高斯内核过滤图像以消除噪声，然后应用大津阈值
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # 5x5高斯内核过滤图像
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('result3.jpg', th3)

    # 调用自定义的函数完成计算
    print('岩石图片:',j)
    calculate('result2.jpg')

    print('-----------------------------------------------------------------------')


