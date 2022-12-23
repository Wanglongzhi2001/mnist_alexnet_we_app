# 基于华为mindspore框架和远程推理框架inferemote的AlexNet手写数字识别的Web应用
![image](https://github.com/Wanglongzhi2001/mnist_alexnet_we_app/blob/main/asset/home.png)
![image](https://github.com/Wanglongzhi2001/mnist_alexnet_we_app/blob/main/asset/result.png)

## 基本功能
- 识别MNIST原数据集图片
- 识别用户自行上传的非MNIST数据集的数字图片(程序会自动灰度化和resize图片)
## 配置
|  深度学习框架   | 远程推理框架  | Web框架  | 前端  |
|  ----  | ----  | ----  | ----  |
|  mindspore1.9.0  | inferemote  | flask2.2.2  | flask-bootstrap3.3.7.1  |
## 识别正确率
![image](https://github.com/Wanglongzhi2001/mnist_alexnet_we_app/blob/main/asset/train_test_acc.png)
## 使用
安装好相关环境后直接运行app.py即可，详见[视频演示](https://www.bilibili.com/video/BV1Wg411b7aF/#reply519184793)


