import numpy as np
import argparse
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image


img_width, img_height = 64, 64

#标签
char_dict= ['上海','上海市','云南','云南省','内蒙古','内蒙古自治区','北京','北京市','台湾','台湾省','吉林','吉林省','四川','四川省','天津','天津市','宁夏','宁夏回族自治区','安徽','安徽省',
            '山东','山东省','山西','山西省','广东','广东省','广西','广西壮族自治区','新疆','新疆维吾尔自治区','江苏','江苏省','江西','江西省','河北','河北省','河南','河南省',
            '浙江','浙江省','海南','海南省','湖北','湖北省','湖南','湖南省','澳门','澳门特别行政区','甘肃','甘肃省','福建','福建省','西藏','西藏自治区','贵州','贵州省','辽宁','辽宁省',
            '重庆','重庆市','陕西','陕西省','青海','青海省','香港','香港特别行政区','黑龙江','黑龙江省']


# 设置参数
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True,
	#help = "Path to the image to be scanned")
args = vars(ap.parse_args())
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while (True):
	# Capture frame by frame
	ret, frame = cap.read()
	# cap.read() 返回一个布尔值（ True/False）。如果帧读取的是正确的，
	# 就是 True。所以最后你可以通过检查他的返回值来查看视频文件是否已经到
	# 了结尾。
	# cap.isOpened()，来检查是否成功初始化了
	gray = cv2.cvtColor(frame, 0)
	# gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', gray)
	if cv2.waitKey(1)&0xFF == ord('q'):
		cv2.imwrite("test.png",frame)    #路径1
		break

cap.release()
cv2.destroyAllWindows()


def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

# 读取输入
image = cv2.imread("test.png")  #路径1
image=cv2.flip(image,1)
#坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()


image = resize(orig, height = 500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# 遍历轮廓
for c in cnts:
	# 计算轮廓近似
	peri = cv2.arcLength(c, True)
	# C表示输入的点集
	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 4个点的时候就拿出来
	if len(approx) == 4:
		screenCnt = approx
		break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)  #路径2
# 展示结果
print("STEP 3: 变换")
#cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(ref, height = 650))
image2=cv2.imread("scan.jpg")  #路径2
# 要被切割的开始的像素的高度值
beH =60
# 要被切割的结束的像素的高度值
hEnd =90
# 要被切割的开始的像素的宽度值
beW = 20
# 要被切割的结束的像素的宽度值
wLen = 73
# 对图片进行切割
dstImg = image2[beH:hEnd,beW:wLen]

# 展示切割好的图片
cv2.imshow("dstImg",dstImg)
cv2.imwrite('dstImg.jpg', dstImg)   #路径3

cv2.waitKey(0)




#加载模型
model = load_model("/content/drive/My Drive/模型/model2.h5")  #模型路径


#生成测试数据流
path="/content/drive/My Drive/省份数据集/验证1"   #路径3
test_datagen = ImageDataGenerator(rescale=1./255)
    # 测试数据只进行重缩放
img = test_datagen.flow_from_directory(
        path,
        target_size=(img_width, img_height),
        batch_size=3,
        color_mode="grayscale",
        shuffle = False,#不打乱顺序
        class_mode=None)#无标签生成


#模型预测
pre=model.predict(img)
#最大概率对应的标签下标
result = np.argmax(pre,axis=1)

print(result)

print(char_dict[result[0]],end='\n')
print(char_dict[result[1]],end='\n')
print(char_dict[result[2]],end='\n')
print("successful")



