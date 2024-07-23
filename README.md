# 基于Yolov10算法和bytetrack的目标追踪的碰撞检测的尝试

车速的计算有很大的问题，和电脑性能以及模型推理速度有关。这个难题极难攻克，也是目前视频车辆测速的难题之一。（大程度依赖于道路情况，摄像头角度等等因素），目前用的是单纯根据识别框位置进行的相对值计算。

**目标检测和轨迹处理已经初步完成**

通过算法直接进行碰撞检测，判断检测框重叠度，轨迹的异常角度变化，速度变化等等。

判断碰撞事故的函数，当速度的变化超过一定阈值，并且角度发生了过大的变化，两者者生成一个针对事故发生率的评分，比如说角度0为0分（无风险），角度45为5（中风险），角度90为7（高风险），超过九十度以此类推，最高十分，三者（速度变化，重叠程度，角度）都有对应的这样的函数，之后对于三种函数赋予权值合成为总的交通事故风险函数，越高越容易发生交通事故，超过阈值就会在窗口打印发生交通事故的信息。

概念阶段：![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/output.gif)

需要解决的问题：
1. 加速度没有归一化
2. 评分函数太粗糙了
3. 计算方式不够科学
4. 框体界面包含更多信息方便调试
5. 尝试直接针对事故本身进行训练将结果与碰撞算法加权后得到最终结果
6. 多帧判断（待定）

评估需要更多的维度使其更科学。

- 仔细研究：https://blog.csdn.net/Kefenggewu_/article/details/123348800

- 数据集：https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs/data
- 数据集：https://universe.roboflow.com/carlos-andres-wilches-perez/cct-drone
- 数据集：





