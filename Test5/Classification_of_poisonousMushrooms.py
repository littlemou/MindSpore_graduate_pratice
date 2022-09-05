import mindspore.dataset as ds # 数据集载入
import mindspore.nn as nn # 各类网络层都在 nn 里面
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor #回调函数
from mindspore.train import Model # 承载网络结构
from mindspore import load_checkpoint # 读取最佳参数
from mindspore import context # 设置 mindspore 运行的环境
from easydict import EasyDict as ed # 超参数保存
import numpy as np # numpy
import matplotlib.pyplot as plt # 可视化
# 文件处理相关
import os
# 华为云文件传输相关
# import moxing
from preprocess import create_dataset
# 这里将网络分为 backbone 和 head，backbone 是 ResNet 包含残差块的部分，head 是最后的全连接层。
from network import resnet50_backbone, resnet50_head
from lr_scheduler import get_lr
from callbacks import TrainHistroy,EvalHistory

# 删除损坏的数据
import os
bad_data = os.path.join('Mushrooms','Russula','092_43B354vYxm8.jpg')
if os.path.exists(bad_data):
 os.remove(bad_data)


import splitfolders
splitfolders.ratio('Mushrooms', output="data", seed=1706, ratio=(.8, .0, .2)) # 这里 Mushrooms 是原始数据文件夹，data 是切分后的数据文件夹

# 最终网络由 backbone 和 head 组成。
class ResNet50(nn.Cell):
    """
    ResNet architecture.
    Args:
    backbone (Cell): ResNet50 backbone 网络
    head (Cell): ResNet50 head 网络
    Returns:
    Tensor, 输出张量
    Examples:
    毒蘑菇图像识别识别实验手册-教师版 第22页
    # >>> ResNet50(resnet_backbone,
    # >>> resnet_head)
    """
    def __init__(self, backbone, head):
        super(ResNet50, self).__init__()
        self.backbone = backbone
        self.head = head
    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

device_target = context.get_context('device_target') # 获取运行装置（CPU，GPU，Ascend）
dataset_sink_mode = True if device_target in ['Ascend','GPU'] else False # 是否将数据通过 pipeline 下发到装置上
context.set_context(mode = context.GRAPH_MODE, device_target = device_target) # 设置运行环境，静态图 context.GRAPH_MODE 指向静态图模型，即在运行之前会把全部图建立编译完毕
print(f'device_target: {device_target}')
print(f'dataset_sink_mode: {dataset_sink_mode}')

# 数据路径
train_path = os.path.join('data', 'train')
test_path = os.path.join('data', 'test')
# 超参数
config = ed({
    # 训练参数
    'batch_size': 32,
    'epochs': 150,

    # 网络参数
    'class_num': 9,
    # 动态学习率调节
    'warmup_epochs': 5,
    'lr_init': 0.01,
    'lr_max': 0.1,
    # 优化器参数
    'momentum': 0.9,
    'weight_decay': 4e-5})

# 创建图像标签列表
category_dict = {0:'Agaricus',1:'Amanita',2:'Boletus',3:'Cortinarius',4:'Entoloma',
 5:'Hygrocybe',6:'Lactarius',7:'Russula',8:'Suillus'}
# 载入展示用数据
demo_ds = ds.ImageFolderDataset(test_path, decode=True)
# 设置图像大小
plt.figure(figsize=(6, 6))
# 打印 9 张子图
i = 1
for dic in demo_ds.create_dict_iterator():
    plt.subplot(3,3,i)
    plt.imshow(dic['image'].asnumpy()) # asnumpy：将 MindSpore tensor 转换成 numpy

    plt.axis('off')
    plt.title(category_dict[dic['label'].asnumpy().item()])
    i += 1
    if i > 9:
        break
plt.show()

train_ds = ds.ImageFolderDataset(train_path, decode=True)
#计算数据集平均数和标准差，数据标准化时使用
tmp = np.asarray( [np.mean(x['image'], axis=(0, 1)) for x in
train_ds.create_dict_iterator(output_numpy=True)] )
RGB_mean = tuple(np.mean(tmp, axis=(0)))
RGB_std = tuple(np.std(tmp, axis=(0)))
print(RGB_mean)
print(RGB_std)

# 训练集
train_data = create_dataset(data_path=train_path,
 mean=RGB_mean,
 std=RGB_std,
 batch_size=config.batch_size,
 usage='train',
 repeat_num=1)
# 测试集
test_data = create_dataset(data_path=test_path,
 mean=RGB_mean,
 std=RGB_std,
 batch_size=config.batch_size,
 usage='test',
 repeat_num=1)

# 训练 step 总数
train_step_size = train_data.get_dataset_size()
# 学习率数组
lr = get_lr(total_epochs=config.epochs,
 steps_per_epoch=train_step_size,
 lr_init=config.lr_init,
 lr_max=config.lr_max,
 warmup_epochs=config.warmup_epochs)

# 网络
backbone_net = resnet50_backbone() # backbone 网络，保存后能提供后续迁移学习使用
head_net = resnet50_head(config.class_num) # head 网络，resnet50 最后的全连接层
net = ResNet50(backbone_net, head_net)
# 损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 优化器
opt = nn.Momentum(net.trainable_params(), lr, momentum=config.momentum,
weight_decay=config.weight_decay)
# 模型
model = Model(net, loss_fn = net_loss,
 optimizer = opt, metrics = {'accuracy','loss'})

time_cb = TimeMonitor(data_size=train_step_size) # 监控每次迭代的时间
loss_cb = LossMonitor() # 监控 loss 值
hist = {'loss':[], 'loss_eval':[], 'acc_eval':[]} # 训练过程记录
# 记录每次迭代的模型准确率
train_hist_cb = TrainHistroy(hist['loss'])
# 测试并记录模型在验证集的 loss 和 accuracy，并保存最优网络参数
eval_hist_cb = EvalHistory(model = model,
 backbone = backbone_net,
 loss_history = hist['loss_eval'],
 acc_history = hist['acc_eval'],
 eval_data = test_data)
cb = [time_cb, loss_cb, train_hist_cb, eval_hist_cb]

model.train(config.epochs, train_data, callbacks=cb)

# 定义 loss 记录绘制函数
def plot_loss(hist):
    plt.plot(hist['loss'], marker='.')
    plt.plot(hist['loss_eval'], marker='.')
    plt.title('loss record')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss_train', 'loss_eval'], loc='upper right')
    plt.show()
    plt.close()
plot_loss(hist)


def plot_accuracy(hist):
    plt.plot(hist['acc_eval'], marker='.')
    plt.title('accuracy history')
    plt.xlabel('epoch')
    plt.ylabel('acc_eval')
    plt.grid()
    plt.show()
    plt.close()
plot_accuracy(hist)

# 使用准确率最高的参数组合建立模型，并测试其在验证集上的效果
load_checkpoint('best_param.ckpt', net=net)
res = model.eval(test_data, dataset_sink_mode=False)
print(res)

# 创建图像标签列表
category_dict = {0: 'Agaricus', 1: 'Amanita', 2: 'Boletus', 3: 'Cortinarius', 4: 'Entoloma',
                 5: 'Hygrocybe', 6: 'Lactarius', 7: 'Russula', 8: 'Suillus'}
ds_test_demo = create_dataset(test_path, mean=RGB_mean, std=RGB_std, batch_size=1, usage='test')


# 将数据标准化至 0~1 区间
def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 设置图像大小
plt.figure(figsize=(10, 10))
i = 1
# 打印 9 张子图
for dic in ds_test_demo.create_dict_iterator():
    # 预测单张图片
    input_img = dic['image']
    output = model.predict(input_img)
    predict = np.argmax(output.asnumpy(), axis=1)[0]  # 反馈可能性最大的类别

    # 可视化
    plt.subplot(3, 3, i)
    input_image = np.squeeze(input_img.asnumpy(), axis=0).transpose(1, 2, 0)  # 删除 batch 维度
    input_image = normalize(input_image)  # 重新标准化，方便可视化
    plt.imshow(input_image)
    plt.axis('off')
    plt.title('True: %s\nPredict: % s'%(category_dict[dic['label'].asnumpy().item()],category_dict[predict]))
    i += 1
    if i > 9:
        break
plt.show()