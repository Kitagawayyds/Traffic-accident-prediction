# 基于Yolov10和BoT-SORT目标追踪的事故检测研究

**拟题目：基于深度学习的实时交通事故检测与风险评估研究**

开始于2024/7/17

ARMS，全称是 "Accident Risk Monitoring System"（事故风险监控系统），是一个基于深度学习的实时交通事故检测（Vision-TAD）和风险评估系统。该系统通过车辆轨迹分析和事故风险计算，能够检测并评估交通事故的发生概率，帮助及时发现潜在的交通事故风险，提供预警信息，提高道路交通安全。

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
## Version 8
在这个版本首先再次优化了显示逻辑，现在更加地清晰，并且更多地参数用于进行细微调整，再次规范了速度的计算方式，以KM/H为单位显示，修正了角度的计算方式，确保轨迹的正确性。

添加了轨迹平滑功能，减少了镜头抖动和碰撞框闪烁带来的影响。

同时编写了映射坐标获取的工具，便于快速处理。

映射的具体逻辑为：

**车道宽度：**
双向四车道是2×7.5米
双向六车道是2×11.25米，
双向八车道是2×15米。

**车道长度：**
- 人行过街斑马线宽45厘米，间距60厘米，长度根据过街人流量确定，常用500厘米和600厘米。
- 双向两车道路面中心线为黄色虚线，用于分隔对向行驶的交通流。一般设在车行道中线上，但不限于一定设在道路的几何中心线上。在保证安全的情况下，允许车辆越线超车或向左转弯。
- 凡路面宽度可划两条机动车道的双向行驶的道路，应划黄色中心虚线。用于指示车辆驾驶人靠右行驶，各行其道，分向行驶。双向两车道路面中心线的划法 黄色虚线长4M，虚线间隔6M，虚线宽15CM。
- 车行道分界线为白色虚线，用来分隔同向行驶的交通流，设在同向行驶的车行道分界线上。在保证安全的情况下，允许车辆越线变换车道行驶。
- 高速公路及城市快速路车道线尺寸虚线长6M，虚线间隔9M，虚线宽15CM。
- 其他道路（城市普通道路）车道线尺寸虚线长2M，虚线间隔4M，虚线宽10-15CM。
- 人行横道线为白色平行粗实践(斑马线)，表示准许行人横穿车行道的标线，人行横道线的设置位置，应根据行人横穿道路的实际需要确定。但路段上设置的人行横道线之间的距离应大于150m。人行横道的最小宽度为3m，并可根据行人数量以1M为一级加宽。
- 禁止超车线，中心黄色双实线。表示严格禁止车辆跨线超车或压线行驶。用以划分上下行方向各有两条或两条以上机动车道而没有设置中央分隔带的道路。本标线为黄色双实线，线宽为15cm，两标线的间隔为15~30cm见线31。除交叉路口或允许车辆左转弯(或回转)路段外，均应连续设置。
- 道路中心双黄线线宽15CM，标线间隔15-30CM
- 停车位尺寸：大的停车位宽4米，长度7米到10米，视车型定。小车车位，宽度2.2米到2.5米，长度5米。旁边道路小车单面停车5米宽，双面6米，大车8米

映射后的效果：

### 正常情况

![display](https://s2.loli.net/2024/08/07/lzgIa4rXNbxtw2n.png)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/test2.gif)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/test2-t.gif)

### 交通事故

![display](https://s2.loli.net/2024/08/07/XeugfdjWkra6AHo.png)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/accident2.gif)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/accident2-t.gif)

这一版本弥补了大量之前版本所遗漏的问题，在接下来的版本，最重要的就是改善角度，速度，重合度的计算，使其更准确，更抗干扰（取多值平均），并且添加针对速度的考量作为新的评估维度（同前面三个），映射至风险评分时也需要更可以解释的科学逻辑。

重写速度方面的逻辑，分为平均速度（使用deque数组），用速度异常波动程度代替加速度。

需要解决的问题：
- **优化代码逻辑，提高推理速度，减少不必要的计算**
- 尝试使用更加轻量的模型
- **风险映射参数以及映射方式需要进行调整升级**
- 继续分析误识别情况的原因
- 针对不同天气情况，时间，车流密度，以及摄像头低像素情况进行鲁棒性数据增强修改
- 碰撞算法改进（增加车辆种类，丰富车辆参数）
- 加速度和角度计算（趋势计算，可以考虑方差等）
- 车流量统计
- **可移植性以及封装（参数化调整）**
- **模块分管化**
- **报错信息处理**
- 添加针对于速度的考量
- 增加测试
- 检查是否有可以用库代替的计算
- imshow显示可选

- 训练数据集（车辆）：https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs/data 注意引用
- 训练数据集（事故）：https://universe.roboflow.com/accident-detection-ffdrf/accident-detection-8dvh5 注意引用
- 测试视频数据集（车流）：https://wayback.archive-it.org/org-652/20231112205116/https:/detrac-db.rit.albany.edu/
- 高质量视频数据（高速）：https://www.vecteezy.com/video/1804377-motorway-with-cars-passing-by
- 数据集：https://github.com/yajunbaby/A-Large-scale-benchmark-for-traffic-accidents-detection-from-video-surveillance?tab=readme-ov-file
- 文章：https://blog.csdn.net/hahabeibei123456789/article/details/103287541 可以了解
- 文章：https://ar5iv.labs.arxiv.org/html/2308.15985
- 文章：https://developer.aliyun.com/article/606837





