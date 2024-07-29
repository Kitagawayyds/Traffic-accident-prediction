# 基于Yolov10算法和bytetrack的目标追踪的事故检测的尝试
## 概念阶段
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

需要解决的问题：
1. 评分函数太粗糙了
2. 计算方式不够科学
3. 代码应该包含更多信息方便调试
4. 多帧判断（待定）
5. 评估需要更多的维度使其更科学。

- 仔细研究：https://blog.csdn.net/Kefenggewu_/article/details/123348800

- 数据集（车辆）：https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs/data 注意引用
- 数据集（事故）：https://universe.roboflow.com/accident-detection-ffdrf/accident-detection-8dvh5 注意引用





