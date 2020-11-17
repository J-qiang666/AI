import cv2  # 机器视觉库，opencv-python
import numpy as np  # 科学计算
import os  # 系统库
from random import shuffle  # 随机数据库
from tqdm import tqdm  # 输出进度库
import matplotlib.pyplot as plt  # 绘图库


plt.rcParams['font.sans-serif'] = ['MicroSoft YaHei']  # 正常显示中文
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 消除警告（TensorFlow binary was not compiled to use: AVX2）
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用第1个GPU来训练模型
train_dir = './input/train/'  # 导入训练数据
test_dir = './input/test/'  # 导入测试数据
img_size = 50  # 统一图片大小
# lr（learn rate），表示学习率
# 如果lr太小，模型很可能需要n年来收敛。如果lr太大，加上不多的初始训练样本，损失可能会极高
# 一般来说，可以尝试0.1、0.01和0.001的学习率
lr = 1e-3


# 将label变成每个类别的概率（类似于哑变量）
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'head':
        return [1, 0]
    elif word_label == 'tail':
        return [0, 1]


# 处理训练数据
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        if (not img.endswith('.jpg')):  # 过滤掉不是以.jpg的图片
            continue
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读入灰度图
        img = cv2.resize(img, (img_size, img_size))  # 将图片变为统一的大小
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    return training_data
train_data = create_train_data()


# 处理测试数据，原理同上
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        if (not img.endswith('.jpg')):
            continue
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    return testing_data


# 导入基于Tensorflow的高级深度学习库tflearn
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d  # 利用2维CNN以及最大采样
from tflearn.layers.core import input_data, dropout, fully_connected  # 输入层，全连接层
from tflearn.layers.estimator import regression  # 复原


# 注意：如果多次运行不同的网络结构图，每次需要先清空图
import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# 两个全连接层与预测层
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')


train = train_data[:-500]
test = train_data[-500:]


X = np.array([i[0] for i in train], dtype=np.float64).reshape(-1, img_size, img_size, 1)
y = np.array([i[1] for i in train], dtype=np.float64)
Xtest = np.array([i[0] for i in test], dtype=np.float64).reshape(-1, img_size, img_size, 1)
ytest = np.array([i[1] for i in test], dtype=np.float64)
model.fit({'input': X}, {'targets': y}, n_epoch=10, validation_set=({'input': Xtest}, {'targets': ytest}),
          snapshot_step=500, show_metric=True, run_id='model')  # 一般设置n_epoch迭代次数60


test_data = process_test_data()
fig = plt.figure()
for num, data in enumerate(test_data[:16]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(4, 4, num + 1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1:
        label = '车尾'
    else:
        label = "车头"
    y.imshow(orig, cmap='gray')
    plt.title(label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('./HeadOrTail.svg')
plt.show()