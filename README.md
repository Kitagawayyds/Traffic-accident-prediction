# 基于Yolov10算法和BoT-SORT的目标追踪的事故检测的尝试
## Concept
当前仅针对对象为普通车辆的事故检测

车速的计算有很大的问题，和电脑性能以及模型推理速度有关。这个难题极难攻克，也是目前视频车辆测速的难题之一。（大程度依赖于道路情况，摄像头角度等等因素），目前用的是单纯根据识别框位置进行的相对值计算。

**目标检测和轨迹处理已经初步完成**

通过算法直接进行事故检测，判断检测框重叠度，轨迹的异常角度变化，速度变化等等。

判断交通事故的函数，当速度的变化超过一定阈值，并且角度发生了过大的变化，两者者生成一个针对事故发生率的评分，比如说角度0为0分（无风险），角度45为5（中风险），角度90为7（高风险），超过九十度以此类推，最高十分，三者（速度变化，重叠程度，角度）都有对应的这样的函数，之后对于三种函数赋予权值合成为总的交通事故风险函数，越高越容易发生交通事故，超过阈值就会在窗口打印发生交通事故的信息。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/concept.gif)

## Version 1
在这个阶段，我引入了新的一个模型用于直接检测事故，并拿得到的置信度与概念版本得到的评分相乘，从而尝试实现更好的结果（使用深度模型直接识别可以捕捉到一些无法量化的细节），代价一是代码更复杂了，二是推理速度更慢了。

绿框是事故模型认为可能发生交通事故的区域，而左上角的红字是最终算法认为事故发生时的提示。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/V1.gif)

## Version 2
在这个阶段着重于对于评分函数的优化，判断逻辑优化，以及信息调试的增加，与此同时增加视频上的信息。

首先是针对加速度，角度以及重叠度的计算方式进行了改进，现在加速度进行了归一化，减少了对于摄像机距离的影响，角度归正在0-180度，不会出现大于180度的情况，重叠度进行了优化：如果一个框非常大而另一个框非常小，重叠率可能会低估碰撞的实际情况。为了解决这个问题，计算两个方向上的重叠率（即每个框相对于另一个框的重叠情况），并取这两个值中的最大值，这样可以更准确地表示碰撞关系。

![display](https://s2.loli.net/2024/07/30/VTzGEoJZshn1xgp.png)

于此同时对评分函数进行了优化，使用非线性函数进行更准确地表达，以达到更合适的评分。

![display](https://s2.loli.net/2024/07/30/XblyrBUPW2kZSds.png)

针对最终的逻辑判断现在使用了最大风险因子法以及使用置信度主导，这样操作不仅提高了置信度发挥的作用，并且使得其余三个参数对于误识别的情况起到缓解作用。

在视频中添加了绿框（事故识别）的置信度以及红框（代表了算法最终认为事故发生的位置）

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/V2.gif)

## Version 3
在这个阶段主要优化代码，调整代码结构以及完善代码，与此同时添加了FPS的显示以判断模型的推理速度。同时对代码控制台的输出进行了控制，以便更好地进行调试。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/V3.gif)

## Version 4
在这个阶段，针对评分计算再次进行了优化，将部分计算矢量化，同时对于细节进行了调整。

于此同时，在速度和角度计算时使用了坐标转换，现在可以将视频中车辆的像素坐标映射为真实坐标（但同样的，这也使得我们在配置代码时需要根据视频实际情况设置映射参数）：

![diaplay](https://s2.loli.net/2024/08/01/zyNSBcAmE5PDWCo.png)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/V4.gif)

需要解决的问题：
1. 多帧判断（待定）
2. 评估需要更多的维度使其更科学。
3. 优化代码逻辑，提高推理速度
4. 尝试使用更加轻量的模型
5. 事故置信度没有关联到车辆而是画面本身，改进逻辑，当检测到事故时才计算速度，角度和重叠率
6. 参数需要进行调整
7. 继续分析误识别情况的原因

- 数据集（车辆）：https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs/data 注意引用
- 数据集（事故）：https://universe.roboflow.com/accident-detection-ffdrf/accident-detection-8dvh5 注意引用
- 文章：https://blog.csdn.net/hahabeibei123456789/article/details/103287541 可以了解





