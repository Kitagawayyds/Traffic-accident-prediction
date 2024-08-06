# 基于Yolov10和BoT-SORT目标追踪的事故检测研究

**拟题目：基于深度学习的实时交通事故检测与风险评估研究**

ARMS系统，全称是 "Accident Risk Monitoring System"（事故风险监控系统），是一个基于深度学习的实时交通事故检测和风险评估系统。该系统通过车辆轨迹分析和事故风险计算，能够检测并评估交通事故的发生概率，帮助及时发现潜在的交通事故风险，提供预警信息，提高道路交通安全。

## Concept
当前仅针对对象为普通车辆的事故检测

车速的计算有很大的问题，和电脑性能以及模型推理速度有关。这个难题极难攻克，也是目前视频车辆测速的难题之一。（大程度依赖于道路情况，摄像头角度等等因素），目前用的是单纯根据识别框位置进行的相对值计算。

**目标检测和轨迹处理已经初步完成**

通过算法直接进行事故检测，判断检测框重叠度，轨迹的异常角度变化，速度变化等等。

判断交通事故的函数，当速度的变化超过一定阈值，并且角度发生了过大的变化，两者者生成一个针对事故发生率的评分，比如说角度0为0分（无风险），角度45为5（中风险），角度90为7（高风险），超过九十度以此类推，最高十分，三者（速度变化，重叠程度，角度）都有对应的这样的函数，之后对于三种函数赋予权值合成为总的交通事故风险函数，越高越容易发生交通事故，超过阈值就会在窗口打印发生交通事故的信息。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/concept.gif)

## Version 1
在这个阶段，我引入了新的一个模型用于直接检测事故，并拿得到的置信度与概念版本得到的评分相乘，从而尝试实现更好的结果（使用深度模型直接识别可以捕捉到一些无法量化的细节），代价一是代码更复杂了，二是推理速度更慢了。

绿框是事故模型认为可能发生交通事故的区域，而左上角的红字是最终算法认为事故发生时的提示。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V1.gif)

## Version 2
在这个阶段着重于对于评分函数的优化，判断逻辑优化，以及信息调试的增加，与此同时增加视频上的信息。

首先是针对加速度，角度以及重叠度的计算方式进行了改进，现在加速度进行了归一化，减少了对于摄像机距离的影响，角度归正在0-180度，不会出现大于180度的情况，重叠度进行了优化：如果一个框非常大而另一个框非常小，重叠率可能会低估碰撞的实际情况。为了解决这个问题，计算两个方向上的重叠率（即每个框相对于另一个框的重叠情况），并取这两个值中的最大值，这样可以更准确地表示碰撞关系。

![display](https://s2.loli.net/2024/07/30/VTzGEoJZshn1xgp.png)

于此同时对评分函数进行了优化，使用非线性函数进行更准确地表达，以达到更合适的评分。

![display](https://s2.loli.net/2024/07/30/XblyrBUPW2kZSds.png)

针对最终的逻辑判断现在使用了最大风险因子法以及使用置信度主导，这样操作不仅提高了置信度发挥的作用，并且使得其余三个参数对于误识别的情况起到缓解作用。

在视频中添加了绿框（事故识别）的置信度以及红框（代表了算法最终认为事故发生的位置）

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V2.gif)

## Version 3
在这个阶段主要优化代码，调整代码结构以及完善代码，与此同时添加了FPS的显示以判断模型的推理速度。同时对代码控制台的输出进行了控制，以便更好地进行调试。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V3.gif)

## Version 4
在这个阶段，针对评分计算再次进行了优化，将部分计算矢量化，同时对于细节进行了调整。

于此同时，在速度和角度计算时使用了坐标转换，现在可以将视频中车辆的像素坐标映射为真实坐标（但同样的，这也使得我们在配置代码时需要根据视频实际情况设置映射参数）：

![diaplay](https://s2.loli.net/2024/08/01/zyNSBcAmE5PDWCo.png)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V4.gif)

## Version 5
使用了更加灵活的存储结构deque来限制数据长度用于长时间检测，同时在评分函数部分进一步优化，减少循环的同时保证算法的正确性，减少了计算量。

于此同时在这个版本添加了针对于碰撞框的理论映射并改进重叠率计算逻辑，不过目前并不成熟，而且计算量巨大。

![display](https://s2.loli.net/2024/08/02/5UEJIpA3K1ar4W7.png)

同时在事故检测部分发生了巨大的变化，如今在事故模型判断出事故后，首先会获取事故框涉及的车辆，并给每辆车赋予事故框的置信度，之后针对每辆车进行计算速度，角度以及重合度，然后将判断为发生事故的车用红框进行标记，虽然算法更加细节了，但是增加了很多的计算量，需要对程序进行优化。

于此同时对于视频处理将之前遗漏的问题进行了修复，现在不再会丢失没有检测到实体的视频帧了。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V5.gif)

## Version 6
在这个版本首先修复了视频分辨率过高导致的问题（尽管我也不认为有人会闲的给摄像头用4K），同时现在可以自定义识别速度（帧数），输出的视频的长度也将与输入视频同步，同时优化了标记事故的方式，更加直观。

更新了overlap的计算，在车辆坐标映射后，根据车辆角度变化，生成更加细节的碰撞框进行计算，同时不再使用椭圆，改回原先的矩形碰撞框。

![display](https://s2.loli.net/2024/08/05/U5YVjlgL9HWMoix.png)

编写了映射可视化代码用于分析碰撞情况

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V6.gif)

## Version 7
首先删除了fps相关的代码，使用tqdm库的进度条来获取程序运行进度以及速度。

统合和修正了之前可视化转换视图的功能(转换情况可以参考`GIF`文件夹下的compare1和compare2)

增添并规范化了控制台的输出，可以更好更直观地判断程序的运行情况

可自定义是否保存推理视频以及显示车辆轨迹。

需要解决的问题：
- **优化代码逻辑，提高推理速度，减少不必要的计算**
- 尝试使用更加轻量的模型
- 参数需要进行调整（针对评分阈值）
- 继续分析误识别情况的原因
- 针对不同天气情况，时间，以及摄像头低像素情况进行鲁棒性数据增强修改
- 碰撞算法改进（增加车辆种类，丰富ab参数）
- 编写获取道路映射坐标的代码
- 可移植性以及封装（参数化调整）
- 模块分管化
- 道路边界环境碰撞逻辑
- 报错信息处理

- 训练数据集（车辆）：https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs/data 注意引用
- 训练数据集（事故）：https://universe.roboflow.com/accident-detection-ffdrf/accident-detection-8dvh5 注意引用
- 测试视频数据集（车流）：https://wayback.archive-it.org/org-652/20231112205116/https:/detrac-db.rit.albany.edu/
- 高质量视频数据（高速）：https://www.vecteezy.com/video/1804377-motorway-with-cars-passing-by
- 文章：https://blog.csdn.net/hahabeibei123456789/article/details/103287541 可以了解





