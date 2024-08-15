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

映射后的效果：

### 正常情况

![display](https://s2.loli.net/2024/08/07/lzgIa4rXNbxtw2n.png)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/test2.gif)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/test2-t.gif)

### 交通事故

![display](https://s2.loli.net/2024/08/07/XeugfdjWkra6AHo.png)

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/accident2.gif)

## Version 9
这一版本接着修复并优化了大量显示类和逻辑类问题，添加了imshow可选功能，与此同时检测事故时过滤掉了轨迹中不稳定的起始值以及最终值并且可选平滑功能，使得轨迹更加符合现实。

修改了关于速度方面的计算方式，现在速度方面着重于速度平均值以及速度的波动量，以这两个变量代替了原先的瞬时加速度，增强了数据的鲁棒性。角度的计算也做了类似的处理，当前获取的是角度变化的平均值。

增加防抖动处理，定义了静止阈值。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V9.gif)

## Version 10
这个版本首先重写了整个风险评分计算的逻辑，更加符合程序运行时的情况。增加了更多的自定义参数方便根据输入实际情况进行调整。

![Figure_1.png](https://s2.loli.net/2024/08/09/3AVaCTYPW4Kp6kf.png)

再者对于映射后代表车辆轨迹的中心点的计算逻辑进行了修正，更贴合车辆本身的运行方式。（详情看研究记录）

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/V10.gif)

## Version 11
这个版本首先增加了针对于轨迹曲率的评估，同时优化了事故检测函数的算法，减少了计算量。增加了轨迹抓取功能，便于更好的分析，同时修改了overlap函数的输入，便于统一输出。

![1.png](https://s2.loli.net/2024/08/12/qLhPT5BxaE36kmG.png)

## Single V1
在这个版本中将事故预测模型删除，只靠事故算法以及车辆检测模型，并大幅度调整代码推理的过程，减少冗余代码，目前取得了优于前面几个版本的计算效率以及识别度，同时这个版本增加了车流量统计功能，当车辆经过了源区域时将被统计进入车流量。今后将主要以这个为主版本进行开发，详细的改动可以看版本11和当前版本的比较。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/singleV1.gif)

## Single V2
在这个版本首先修复了轨迹显示问题，其次对代码进行了封装，可以直接在命令台上设置运行参数了。关于映射区域的参数和放大系数现在直接通过`transform.txt`文件进行读取，更加便捷。添加了报错处理，可以更好地发现问题。

同时，针对于碰撞框生成，现在将根据检测对象的具体类型进行生成，生成的具体参数依照参数文件`spec.txt`，其中分为else和非else参数，非else函数包含了主要关注的对象。

![ 2024-08-13 162411.png](https://s2.loli.net/2024/08/13/mDkAz2l13NtaFPE.png)

## Single V3
在这个版本首先调整了针对于镜头抖动导致的重叠度误判问题，其次在映射视频中增加了非常多的信息以便更直观地进行分析。

![display](https://github.com/Kitagawayyds/Traffic-accident-prediction/blob/main/gif/singleV3.gif)

## ARMS V1
此版本为正式版，为程序添加了前端，以便更好地操作，同时添加了实时检测模式，更符合实际应用中的情况。详情请看代码。
运行方式：
```shell
streamlit run ARMSV1.py
```
至此整个程序已趋近于完善，下面的工作主要以查找bug和逻辑错误，完整性检查，封包解耦，以及报错处理，**前端制作**为主。

增加映射视频的信息，比如id以及类型，或者碰撞显示

需要解决的问题：
- 优化代码逻辑，提高推理速度，减少不必要的计算
- 尝试使用更加轻量的模型
- **风险映射参数以及映射方式需要进行调整升级**
- 继续分析误识别情况的原因
- 针对不同天气情况，时间，车流密度，以及摄像头低像素情况进行鲁棒性数据增强修改
- 可移植性以及封装（参数化调整）
- 模块分管化
- **报错信息处理**
- **增加测试**
- 检查是否有可以用库代替的计算
- 生成requirement.txt


## 参考资料

- 车辆跟踪识别尝试使用3D预测模型，https://docs.ultralytics.com/datasets/detect/argoverse/
- 训练数据集（车辆）：https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs/data
- 训练数据集（事故）：https://universe.roboflow.com/accident-detection-ffdrf/accident-detection-8dvh5
- 训练数据集（事故）：https://universe.roboflow.com/ambulance-0rcqn/accident_detection-trmhu
- 测试视频数据集（车流）：https://wayback.archive-it.org/org-652/20231112205116/https:/detrac-db.rit.albany.edu/
- 高质量视频数据（高速）：https://www.vecteezy.com/video/1804377-motorway-with-cars-passing-by
- 数据集（事故测试）：https://github.com/yajunbaby/A-Large-scale-benchmark-for-traffic-accidents-detection-from-video-surveillance?tab=readme-ov-file
- 3D数据集：https://thudair.baai.ac.cn/roadtest
- yolov10：https://arxiv.org/pdf/2405.14458
- 文章：https://blog.csdn.net/hahabeibei123456789/article/details/103287541
- 文章：https://blog.csdn.net/qq_39523365/article/details/129733150
- 文章：https://ar5iv.labs.arxiv.org/html/2308.15985
- 文章：https://developer.aliyun.com/article/606837
- 文章：https://arxiv.org/pdf/2407.17757
- 文章：https://arxiv.org/pdf/2002.10111
- 文章：https://www.jiqizhixin.com/articles/2019-08-01-13
- 文章：https://arxiv.org/pdf/1612.00496
- 文章：https://sjcj.nuaa.edu.cn/html/2018/2/20180220.htm
- 项目：https://github.com/HuangJunJie2017/BEVDet
- 项目：https://github.com/skhadem/3D-BoundingBox
- 项目：https://github.com/ruhyadi/yolo3d-lightning
- 项目：https://github.com/zhangyp15/MonoFlex/tree/main
- 项目：https://github.com/maudzung/Complex-YOLOv4-Pytorch
- streamlit库使用：https://blog.csdn.net/weixin_44458771/article/details/135495928

## 研究记录

### 映射的具体标准

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

### 映射中存在的误差
在现在的映射逻辑中，会出现由立体图形高度缺失所带来的映射坐标误差：

![display](https://s2.loli.net/2024/08/08/qzXAHn3reK6G4xw.png)

如图所示，可以将`ABCD-A'B'C'D'`看作是三维空间中车辆实际的形态，而红框则是车辆识别算法中对车辆标定的框，当前算法取得了红框中心点`G`作为映射依据得到了右图经过`viewtransfomer`获得的映射坐标，而真正能代表车辆映射依据中心点的其实是`BCC'B'`的中心`G`，而按照算法的映射，其转换后对应的坐标其实是`N`，可以发现，两者的误差不小，可能会对事故检测造成影响，因此需要将`G`的坐标转换为`H`。但是由于无法将车宽，车高，车长等参数动态转换为视图像素映射值，当前不可能进行该映射。

除此以外，还存在视觉误差问题，即`BC`和`B'C'`在图像中并非等长，这里的误差是视距带来的（想象一下等待火车进站），但因为过小，且计算成本过大，在这里不再考虑，近似做相等。

其他设备问题在这里也不再考虑。

常见针对该问题的方法，用于单目检测的BEVdet，yolo3D，SMOKE，MonoDLE等等，大致方向是基于2D的模型增加深度感知层，使用像nuscense或者argoverse这种雷达扫描的点云类型的数据集进行训练。不过推理速度过慢，而且现阶段以我的知识储备和代码能力有难度直接修改模型架构，需要大量时间，因此通过数学角度进行了优化：

![1.png](https://s2.loli.net/2024/08/09/FwJ5qmoMxb69A1l.png)

![2.png](https://s2.loli.net/2024/08/09/DkYvjmxbNPOdcHJ.png)

![3.png](https://s2.loli.net/2024/08/09/ALMHr6ySfWBmVh1.png)

从上图可以看出，`J`和`F`的误差基本可以接受，而`j`的计算方式可以全凭检测框一致的内容进行计算，通过这种改进可以获得车底盘的近似代表点用于转换轨迹。

### 轨迹的不均衡性
因为轨迹的计算方式取决于检测框所计算的点，因此在车刚出现以及消失的时间段中这种轨迹的计算其实是不准确的，会影响速度，细微的角度：

![display](https://s2.loli.net/2024/08/09/gQwmi91PCHefTBq.png)

要想办法将这种影响消除。

为了获取更稳定的轨迹，基本考虑三种情况：
- 先平滑后转换
- 先转换后平滑
- 先平滑后转换再平滑
简单的对比一下:

![Figure_1.png](https://s2.loli.net/2024/08/12/u8PtnyAOQIc7D3B.png)

在single V1版本以后开始采用双平滑

除此以外，在识别远距离车辆以及车辆在部分遮挡的的时候轨迹依旧会出现问题：

![Figure_1.png](https://s2.loli.net/2024/08/12/E92IOx4WQz6nfP5.png)

![Figure_2.png](https://s2.loli.net/2024/08/12/Elv3ezFObZaDRPM.png)

### 映射至风险评分的科学逻辑

1. 角度变化与速度的关系：
车辆速度为0时：当车辆速度为0时，车辆处于静止状态，理论上角度变化也应该为0，因为车辆不移动，不会产生角度变化。
车辆速度非0时：车辆在行驶时，角度变化通常由车辆的转向操作或行驶路径的弯曲导致。车辆的速度越快，角度变化的速率可能也会受到影响，但角度变化的幅度更多地取决于转向角度和道路曲率。
速度与速度波动的关系：

2. 速度与速度波动的关系：
速度波动：速度波动通常表示车辆速度的变化率。车辆的速度波动可以受到驾驶行为（如加速和刹车）、道路条件以及交通状况的影响。
车辆速度高时：当车辆速度较高时，速度波动可能会更加明显，因为小的加速或减速会对速度变化产生较大的影响。
车辆速度低时：低速行驶时，速度波动可能较小，车辆的加速和减速对总速度的影响比较平稳。
角度变化与速度波动的关系：

3. 角度变化与速度波动的关系：
高速行驶时：车辆在高速行驶时，如果发生较大的角度变化（如急转弯），速度波动可能会加大。这是因为急转弯会使车辆的侧向力增加，从而影响行驶的稳定性和速度。
低速行驶时：在低速情况下，角度变化对速度波动的影响相对较小，因为低速下转弯的幅度和频率通常较低。
车辆之间重叠程度的影响：

4. 车辆之间重叠程度的影响：
车辆重叠：车辆之间的重叠程度通常与车辆之间的距离和相对速度有关。较大的重叠可能表示车辆之间的距离较近，且相对速度较高。
影响角度变化：车辆之间的重叠可能会影响车辆的转向角度，尤其在密集交通条件下，车辆可能需要更频繁地调整方向。
影响速度波动：在车流量大的情况下，车辆之间的距离较近，可能导致频繁的加速和减速，从而加大速度波动。

5. 速度和轨迹曲率的关系：
一般来说车辆在越快的情况下，曲率越大所代表的危险也就越大，因此在高速行驶时车辆对于微小的水平位移尤其敏感。
