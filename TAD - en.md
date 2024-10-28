# Research on Real-Time Traffic Accident Detection and Risk Assessment Based on Deep Learning

## Abstract

​	This research presents the Accident Risk Monitoring System (ARMS), a real-time traffic accident detection and risk assessment system. ARMS leverages advanced deep learning techniques to analyze vehicle trajectories and assess the risk of potential accidents, thereby significantly reducing false alarm rates. The system provides effective early warning information for road traffic safety by evaluating factors such as vehicle speed change, trajectory angle variation, curvature, and overlap degree. Through extensive experiments, ARMS demonstrates improved accuracy and real-time capabilities in traffic accident detection, offering valuable support for traffic management and safety enhancements.

## Keywords

Deep learning; Trajectory analysis; Risk assessment; Real-time monitoring; Collision warning

## Introduction

### Research Background

​	With the acceleration of urbanization, road traffic safety has gradually raised global attention and concern. According to the *Global Road Safety Status Report 2023* [1], road traffic safety remains a serious issue worldwide. Approximately 1.19 million people die in road traffic accidents every year, although a decrease from 2010, still represents a major cause of death. The death rate is particularly high among motorcyclists and automobile passengers, and over 90% of traffic fatalities occur in low- and middle-income countries. These accidents not only bring great grief to the victims and their families but also impose a significant economic burden on society. Therefore, effectively preventing and reducing traffic accidents and improving road efficiency has become a pressing global issue.

​	Traditional traffic accident detection methods, such as closed-circuit television (CCTV) monitoring systems, primarily rely on manual monitoring and alarm systems [2]. These methods have obvious limitations, including long response times, high false alarm rates, and insufficient reliability. For example, a study on urban traffic monitoring systems found that the average response time for manual monitoring systems exceeded 30 seconds, with a false alarm rate of up to 40% [3]. With the rapid development of deep learning technology, traffic accident detection methods based on computer vision have attracted much attention due to their high accuracy and real-time performance. These methods analyze road surveillance videos to automatically detect and predict potential traffic accidents, so that provide decision support for traffic management departments and safety warnings for road users. A comparative study showed that traffic accident detection systems using deep learning algorithms improved accuracy by approximately 20% compared to traditional methods, and reduced response times by 75% [4].

### Literature research

​	In the field of traffic accident detection technology, researchers are progressively advancing the exploration of detection methods under complex traffic scenarios. J. Fang and colleagues provided a comprehensive overview of the evolution of vision-based traffic accident detection and prediction methods, emphasizing the transition from traditional methods relying on manual features to deep learning technologies (Fang et al., [5]). They noted that despite significant progress in accident detection through deep learning, challenges remain in handling complex scenarios such as occlusions and lighting changes. Existing vision-based accident detection systems experience a significant drop in accuracy in low-light or heavily occluded scenes, primarily due to the limited learning capacity of deep learning models for sparse data, leading to a decline in predictive performance under extreme conditions.

​	To enhance data quality, Bemposta Rosende and colleagues developed a set of traffic image datasets captured from drones, aiming to provide high-quality traffic management data for training machine vision algorithms, particularly in the field of road traffic monitoring and accident detection (Rosende et al., [6]). Although these datasets offer a rich perspective for accident detection, their generalization ability under various weather conditions, especially during adverse weather like rain and snow, still needs improvement, as detection performance is significantly reduced under such conditions.

​	In terms of improving detection accuracy, Arsalan Mousavian and colleagues proposed a 3D bounding box estimation method that combines deep learning and geometry (Mousavian et al., [7]). This method relies not only on the image features extracted by convolutional neural networks but also uses geometric constraints to enhance the accuracy of 3D object detection. Although the method performed well in experiments, the accuracy of bounding box estimation significantly decreases in practical applications when target objects are occluded or partially overlapped.

​	To better handle spatiotemporal information, Wang Chen and colleagues proposed an improved dual-stream network for vision-based traffic accident detection (Chen et al., [8]). The model combines spatiotemporal information to achieve more accurate accident detection effects. Although the model has shown good performance in accident detection tasks, its detection accuracy still needs improvement when dealing with dynamically complex scenes, especially in high-speed scenarios where the ability to capture details is limited.

​	Addressing the need for real-time processing, Li Haitao and colleagues proposed a real-time traffic accident automatic detection method based on temporal convolutional autoencoders (Li et al., [9]). This method automatically detects traffic events in video streams and has strong real-time processing capabilities, suitable for practical application scenarios in traffic management. However, the model still needs to improve its detection accuracy in complex traffic environments, such as scenes with occlusion or significant lighting changes, especially in high-speed dynamic scenes.

​	Lastly, Lyu Pu and colleagues developed a traffic accident severity prediction model based on deep residual and attention mechanisms (Lyu et al., [10]). The model effectively predicts the severity of accidents by combining the latest deep residual networks and attention mechanisms. However, the model's generalization ability is limited, especially when facing rare types of accidents, its predictive performance is not ideal.

​	Overall, these studies reflect the gradual deepening of efforts to improve the performance of traffic accident detection systems, with researchers continuously addressing technical challenges to achieve higher accuracy and robustness. These advancements provide a solid foundation and new challenges for subsequent research, including our proposed real-time traffic accident detection and risk assessment system (ARMS). Building on these existing studies, our system further explores, especially in trajectory analysis and collision warning, aiming to provide more precise and timely traffic accident detection and risk assessment.

### Significance of the Study

​	This study is of paramount importance due to the development of a state-of-the-art real-time traffic accident detection and risk assessment system, ARMS, which harnesses the power of deep learning algorithms. Designed to analyze road surveillance videos in real-time, ARMS is capable of accurately identifying traffic accidents and assessing their risk levels. The practical implications of this research are profound, as it is designed to enhance traffic management efficiency, reduce the frequency of traffic accidents, and ultimately protect human lives and property. Furthermore, this work is set to enrich the field by providing new practical examples and theoretical foundations for the application of deep learning in intelligent transportation systems.

​	The essence of our research's innovation lies in the sophisticated integration of deep learning with traffic accident detection and risk assessment. This integration is meticulously crafted to surpass the limitations of current technologies, particularly in managing complex scenarios and ensuring real-time performance. Central to ARMS is the synergy between the YOLOv10 object detection algorithm and the BoT-SORT tracking algorithm, which together achieve high-precision vehicle detection and trajectory tracking. To bolster the accuracy of accident detection, we introduce a novel perspective transformation method that maps vehicle pixel coordinates from the image plane to real-world coordinates. Complementing this is a pioneering risk score calculation method that evaluates vehicle motion characteristics such as speed change, trajectory angle variation, and overlap degree between vehicles. Through rigorous experimentation, we have fine-tuned our algorithms and system parameters, ensuring ARMS's adaptability across diverse traffic environments and scenarios.

​	As we progress through this paper, we will unravel the intricate design and implementation of ARMS, shedding light on its system architecture, data processing techniques, and other key elements that culminate in its cutting-edge capabilities. This comprehensive exposition will demonstrate how ARMS transcends traditional boundaries, offering a robust solution for traffic accident detection and risk assessment in the realm of intelligent transportation systems.

## System Design

​	This section delves into the architecture and implementation details of ARMS, focusing on its core components and data processing sequence. It's structured to provide a clear understanding of how ARMS functions in real-time to detect and assess traffic accidents.

![sys](C:/Users/kitag/Desktop/learning/%E8%81%94%E9%82%A6%E5%AD%A6%E4%B9%A0/%E8%AE%BA%E6%96%87%E5%87%86%E5%A4%87SCI/%E5%9B%BE/sys.png)

​	*Figure 1 presents the system architecture of ARMS, showcasing its modular design and workflow. The system initiates by capturing data through either a video camera or a video file, which then proceeds to the Core Processing Module for vehicle detection using YOLOv10 and tracking via Bot-SORT. Following this, Gaussian Filtering smooths the data, and Perspective Transformation converts pixel coordinates into real-world coordinates. The system evaluates vehicle motion characteristics like speed and trajectory to compute a risk score, which is then visualized alongside annotated video output. This comprehensive approach allows ARMS to provide accurate real-time traffic accident detection and risk assessment, enhancing traffic safety management.*

### Overview of System Architecture

​	ARMS adopts a modular design, controlled by a front-end interface and composed of three main modules: data input, core processing, and data output. This modular design ensures system extensibility and decoupling. The primary goal of the system is to achieve real-time detection and prediction of traffic accidents through object tracking technology.

1. **Data Input Module**: This module handles video file input, real-time camera data capture, and user-configured parameter files. Users can upload necessary data and configurations, such as vehicle detection models, video files, perspective transformation parameters, and vehicle specification files through the Streamlit interface.

2. **Core Processing Module**: This is the core of the system, responsible for vehicle detection and tracking, trajectory smoothing, perspective transformation, speed and curvature calculation, risk assessment, and accident detection. The system processes video frames to identify vehicle positions, track their motion, and calculate risk scores.

3. **Data Output Module**: It presents and saves the processing results in various formats, including real-time trajectories, risk scores, accident detection results, and annotated images or videos.

### Data Processing Workflow

​	The data processing workflow of ARMS is divided into four key stages, each of which is critical to the final accident detection result:

#### Data Input

​	The system supports two data input modes: video file upload and real-time camera capture. Users need to upload the perspective transformation file and vehicle specification file for accurate analysis by the system.

Below are some details about the surveillance data:

1. According to the *Technical Specifications for Road Traffic Safety Violation Image Evidence Collection* [23], the image resolution for digital imaging equipment should not be lower than 1280×720 pixels, i.e., about 9.2 million pixels. However, some high-end surveillance cameras may provide higher resolutions, with a maximum resolution of 3392×2008 pixels.

2. The frame rate of surveillance cameras is typically set to 25fps or 30fps, which is sufficient to capture clear and smooth dynamic images. For example, some smart traffic cameras have frame rates as high as 25fps.

3. For 720P cameras, the typical bit rate is 2M, but for road surveillance scenarios, a 4M bit rate may be necessary to accommodate fast-moving traffic and large scene changes.

​	In the YOLOv10 algorithm, input image sizes are typically resized to 640×640 pixels for detection.

#### Vehicle Detection and Tracking

​	In ARMS, the YOLOv10 [11] and BoT-SORT [12] algorithms are carefully selected to ensure the efficiency and accuracy of vehicle detection and tracking. YOLOv10, with its innovative no-NMS training strategy and optimized model architecture, significantly enhances object detection performance while reducing computational costs. The BoT-SORT algorithm excels in multi-object tracking, particularly in handling complex scenarios with stability and accuracy. The combination of these two algorithms provides robust support for real-time traffic accident detection and prediction in ARMS.

#### Trajectory Transformation and Smoothing

​	Vehicle trajectory data undergo perspective transformation and smoothing to improve detection accuracy. The transformation process converts pixel coordinates into real-world coordinates.

####  Risk Assessment and Incident Detection

​	The system calculates a risk score based on the vehicle's trajectory data and determines whether an incident has occurred based on a pre-programmed risk threshold. The calculation of the risk score takes into account factors such as speed fluctuation, angle change, trajectory curvature, and overlap degree.

### User Interface Design

​	The user interface of ARMS is based on the Streamlit framework, providing an intuitive operation process and real-time feedback functionality.

![img](https://s2.loli.net/2024/09/11/4xeIrNc8gMPKn3m.png)

​	*Figure 2 displays the frontend interface of ARMS, which is a user-friendly dashboard that facilitates the configuration and operation of the traffic accident detection system. It allows users to upload video files, select vehicle specifications, and set output directories for results. Key parameters such as the target FPS and risk threshold can be adjusted through this interface, ensuring the system's responsiveness to various traffic conditions. The dashboard also displays real-time detection results, including speed, angle change, and overlap metrics, providing immediate insights into potential accident risks. For further information and updates, users are directed to the program's website. This frontend is essential for interacting with ARMS, making the process of traffic monitoring and risk assessment more accessible and efficient.*

1. Users can configure system parameters through the sidebar, such as selecting analysis modes, uploading necessary files, and adjusting risk parameters.

2. The system dynamically displays real-time video frames with annotations on the main interface, including the vehicle's ID, category, trajectory, and risk score. Additionally, it provides a curve chart for incident detection and a vehicle risk matrix table.

3. Users can choose to save the inference video, incident frames, and detailed matrix data, which can be used for subsequent analysis and documentation.

## Methodology

​	This section will introduce the core algorithms of this study, including the mathematical model for perspective transformation, extraction of vehicle physical characteristics, various risk scoring functions, and the comprehensive risk calculation formula. In addition, optimization measures are proposed to improve the computational efficiency of the system and the accuracy of incident detection, laying the foundation for the next step of experiments and result analysis.

### Innovative Perspective Transformation Method

![image-20240913141146215](https://s2.loli.net/2024/09/13/1igwPLrYjWHEB8c.png)

​	*Figure 3 shows the geometric relationships for perspective transformation, detailing the calculation of key points K, A, and C within the detection box. This process is crucial for estimating the actual position of a vehicle's chassis and enhancing the accuracy of traffic accident detection.*

​	In the perspective transformation process, the input includes the source area, target area, and target detection box. The source area is the sample area used for transformation, while the target area represents its actual size in the real world. In this study, the coordinate unit of the source area is pixels, and the coordinate unit of the target area is meters.

| Parameter                      | Description                                 | Size                                                         |
| :----------------------------- | :------------------------------------------ | :----------------------------------------------------------- |
| Lane Width                     | Two-way four-lane                           | 2×7.5 meters                                                 |
| Lane Width                     | Two-way six-lane                            | 2×11.25 meters                                               |
| Lane Width                     | Two-way eight-lane                          | 2×15 meters                                                  |
| Pedestrian Crossing            | Width                                       | 45 cm                                                        |
| Pedestrian Crossing            | Spacing                                     | 60 cm                                                        |
| Pedestrian Crossing            | Length                                      | 500 cm or 600 cm                                             |
| Road Centerline                | Two-way two-lane                            | Yellow dashed line, 4 meters long, 6 meters apart, 15 cm wide |
| Carriageway Boundary Line      | Same direction travel                       | White dashed line, 2 meters long, 4 meters apart, 10-15 cm wide (urban ordinary roads) |
| Carriageway Boundary Line      | Highway and Urban Expressway                | White dashed line, 6 meters long, 9 meters apart, 15 cm wide |
| Pedestrian Crossing Line       | Width                                       | Minimum 3 meters, can be widened as needed                   |
| No Overtaking Line             | Center yellow double solid line             | Line width 15 cm, 15-30 cm apart                             |
| Road Center Double Yellow Line | Line width                                  | 15 cm, 15-30 cm apart                                        |
| Parking Space Size             | Large vehicle                               | Width 4 meters, length 7-10 meters                           |
| Parking Space Size             | Small vehicle                               | Width 2.2-2.5 meters, length 5 meters                        |
| Parking Space Size             | Side road single parking for small vehicles | Road width 5 meters                                          |
| Parking Space Size             | Side road double parking for small vehicles | Road width 6 meters (small vehicle) / 8 meters (large vehicle) |

​	*Table 1 lists the configurations used by the author to estimate the parameters of the mapping area, providing detailed size information for lane width, pedestrian crosswalk, road centerline, carriageway boundary line, pedestrian crossing line, no overtaking line, road center double yellow line, and parking space size.*

​	In the general mapping logic, the center point $$K$$ can be calculated from the information of the four vertices of the detection box [34], which is used as the basis for the projection transformation. However, since vehicles are three-dimensional solid objects, and the projected detection box is only an approximation on a two-dimensional plane, there will be errors in the mapping process, losing the depth of the car.

​	As shown in Figure 3, $$EFGH-E'F'G'H'$$represents the actual shape of the vehicle in three-dimensional space, while the detection box is only a two-dimensional rectangular box (red box) identified by the algorithm. If the center point $$K$$ of the red box is used as the basis for coordinate mapping, after the *viewTransformer* transformation, the mapped coordinate $$O$$ in the right diagram is obtained, which has a large error from the actual representative point $$N$$ of the vehicle and cannot accurately represent the true position of the vehicle.

​	The correct mapping reference point should be the center point $$A$$ of $$FGG_1F_1$$, as it is closer to the actual projection of the vehicle on the chassis $$N$$. However, in the current algorithm, the model does not have a depth perception network, so we cannot obtain the parameters such as the width, height, and length of the vehicle and convert them into accurate pixel mapping values, which leads to errors.

​	In addition to the errors caused by depth information, there are also visual errors. For example, in Figure 3, $$FG$$ and $$F_1G_1$$ are not equal in length. This kind of error is caused by the scaling effect of the viewing distance, but the impact of these errors in this problem is small and can be ignored.

​	To reduce this mapping error, one method is to obtain the depth information of the vehicle, using monocular 3D object detection schemes such as BEVDet [14], 3D-BoundingBox, MonoFlex [15], etc. These methods recover the three-dimensional information of the vehicle from two-dimensional images by adding a depth perception layer and using multimodal datasets in the field of autonomous driving such as NuScenes [16], Argoverse [17] for training. However, these models are high computational costs, low calculation speed, and low detection accuracy during inference procedure.

![image-20240913141920912](https://s2.loli.net/2024/09/21/85gyouX1OUrYiZt.png)

​	*Figure 4 illustrates the method used to estimate the actual position of a vehicle's chassis projection within the detection box by utilizing a reference angle. The angle assists in minimizing errors and provides a more accurate representation of the vehicle's position in the detection box, which is crucial for the system's trajectory analysis and risk assessment capabilities.*

​	Another method proposed by the author, inspired by the work of S. B. Kang et al. [33], provides a simplified way to estimate the approximate representative point of a vehicle's chassis. This is achieved by estimating the position of the chassis' center point using the geometric relationship of the detection box, thereby reducing error. Typically, cameras installed on the road have an average overhead angle of 30°, which offers more detailed information and a better field of view. In this case, we assume that as the vehicle drives in front of the camera, the average angle between the vehicle and the camera is 30°. To minimize error, we use this angle as a reference to estimate the actual position represented by the chassis projection within the detection box (Figure 4).

​	The calculation formula for the point $$C$$ (approximately two-thirds down from the top of the detection box, influenced by the vehicle's parameters) is as follows:
$$
C = \left( \frac{X_1 + X_2}{2}, \frac{Y_1 + 2 \times Y_2}{3} \right)\quad\quad\quad (1)
$$
​	As shown in Figure 3, the error between the mapped point $$C$$ and the points $$P$$ and  $$N$$ is acceptable. Therefore, with this improved method, we can obtain an approximate representative point of the vehicle’s chassis, which can further be used for trajectory transformation and risk assessment.

​	This mathematical optimization method only requires the geometric information of the detection box, without relying on 3D reconstruction, significantly simplifying the computational complexity. Although it cannot entirely eliminate errors, it significantly reduces the impact of errors on accident detection while maintaining the efficiency of the inference process. After multiple comparisons between the actual vehicle positions and the mapped vehicle positions obtained using both methods, the mapping coordinates obtained using this estimation method are 47% more accurate than those calculated based on the center point.

​	The process of converting the representative point $$(x, y)$$ from the 2D vehicle detection box to real-world coordinates can be achieved by the projection transformation formula. This formula maps the pixel coordinates $$(x, y)$$ on the image plane to the real-world coordinates $$(X, Y)$$, thereby converting the vehicle’s position in the image to its position in the actual scene.

​	The projection transformation formula is as follows: 
$$
X = \frac{a_1x + a_2y + a_3}{c_1x + c_2y + c_3}, \quad Y = \frac{b_1x + b_2y + b_3}{c_1x + c_2y + c_3}\quad\quad\quad (2)
$$
Where:

1. $$(x, y)$$ are the pixel coordinates in the image, estimated by the mathematical method in the previous step.

2. $$(X,Y)$$ are the transformed 2D plane coordinates in the real world, typically representing the vehicle's position in the actual scene.

3. $$a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3$$ are the parameters of the perspective transformation matrix, which are the elements of the transformation matrix calculated by the *cv2.getPerspectiveTransform* function using the source and target areas as inputs.

### Collision Box Calculation

​	By using the above projection transformation formula, we convert the pixel coordinates in the image to the position in the actual scene, thus obtaining the vehicle's geometric center $$(X, Y)$$. Based on the output provided by the vehicle detection model, we can obtain the type of vehicle and acquire the vehicle's width and length from the corresponding vehicle parameter configuration file. The vehicle's direction is obtained from the *current_angle* of the *calculate_angle* function: using the two most recent points of the trajectory to obtain the absolute angle of the vehicle. With the vehicle's geometric center, $$(X, Y)$$ length, width, and direction, we can generate a rectangular collision box. This box can be adjusted from a standard position and orientation to the actual position and orientation of the vehicle through rotation and translation operations.

The four vertices of the collision box $$V_i\quad(i=1,2,3,4)$$ can be calculated as follows: 
$$
V_i = \left( X + \frac{L}{2} \cdot \cos(\theta + \frac{\pi}{2}(i-1)), Y + \frac{L}{2} \cdot \sin(\theta + \frac{\pi}{2}(i-1)) \right)\quad\quad\quad (3)
$$

1. Suppose that the length of the vehicle is $$L$$, the width is $$W$$, and the orientation is $$\theta$$ (angle with the horizontal).

2. The geometric centre of the vehicle is $$(X, Y)$$.

3. Here, $$\cos(\theta + \frac{\pi}{2}(i-1))$$ and  $$\sin(\theta + \frac{\pi}{2}(i-1))$$calculate the displacement of each vertex relative to the vehicle center in the $$x$$ and $$y$$ directions, respectively.

4. $$i = 1, 2, 3, 4$$ correspond to the four vertices of the collision box.

​	Through the above transformations, the actual trajectory of the vehicle can be used for the extraction of vehicle physical characteristics, risk assessment, and the calculation of the probability of collisions between vehicles, combining indicators such as trajectory curvature and overlap degree.

### Extraction of vehicle physical properties

#### Velocity change analysis

​	Velocity is calculated by displacement change between frames as:

$$
v = \Delta d\times FPS \times 3.6\quad\quad\quad (4)
$$

​	Where $$ v $$ denotes the speed of the car (km/h), $$ \Delta d$$ is the displacement distance between the frame and the previous frame, and the theoretical speed of the car can be obtained by multiplying the number of frames and the km/h conversion factor on this basis.

​	In order to reduce the interference caused by abnormalities such as jitter in the detection frame, we calculate the average value of the speed between all frames as a reference value. The average speed $$\bar{v} $$ is calculated as:

$$
\bar{v} = \frac{1}{N} \sum_{i=1}^{N} v_i\quad\quad\quad (5)
$$

​	where $$v_i $$ is the velocity of the $$i $$th frame and $$N $$ is the total number of frames.

​	Next, we compute the velocity fluctuation, which is used to reflect the degree of stability or variation of the vehicle's velocity. Velocity fluctuation is expressed by calculating the standard deviation of the velocity, which is given by:

$$
\sigma_v = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (v_i - \bar{v})^2}\quad\quad\quad (6)
$$

where:

1. $$\sigma_v $$: the standard deviation of velocity fluctuations, indicating the degree of dispersion of velocity changes.

2. $$v_i $$: the velocity of the $$i $$th frame.

3. $$\bar{v} $$: the average velocity.

4. $$N $$: total number of frames.

​	With this analysis, we get a more intuitive picture of the velocity fluctuations of the vehicle throughout the trajectory, larger fluctuations implying more erratic changes in velocity. The magnitude of the velocity will also affect the subsequent estimation of the risk score.

#### Anomalous Angular Changes in Trajectories

​	The relative change in angle is calculated by the angle between three adjacent points. Specifically, we first calculate the difference in angle between the first two points and the last two points, reflecting the curvature or change in direction of the trajectory. The angle between two points $$ \theta $$ is calculated by the following formula:

$$
\theta = \arctan\left(\frac{y_2 - y_1}{x_2 - x_1}\right)\quad\quad\quad (7)
$$

​	where $$ (x_1, y_1) $$ and $$ (x_2, y_2) $$ denote the coordinates of two neighbouring points in the trajectory, respectively.

​	In order to measure the angular change, the difference in angle between every three neighbouring points is calculated and the angular change is obtained as:

$$
\Delta \theta = |\theta_2 - \theta_1|\quad\quad\quad (8)
$$

​	Also, to ensure that the angular change does not exceed 180 degrees, the calculated angular change values are restricted to values between 0 and 180 degrees, i.e.:

$$
\Delta \theta = \min(\Delta \theta, 360 - \Delta \theta)\quad\quad\quad (9)
$$

​	Similarly, in order to reduce the interference caused by anomalies such as jitter in the detection frame, we calculate the angular change between a number of recent points (e.g., the last 5 points) and take the average value, called the average angular change, which is given by:

$$
\bar{\Delta \theta} = \frac{1}{N} \sum_{i=1}^{N} \Delta \theta_i\quad\quad\quad (10)
$$

​	where $$ N $$ denotes the number of angular changes calculated and $$ \Delta \theta_i $$ is the angular change between each set of three points.

​	In addition, we calculate the current angle of the last segment of the trajectory, which is used to indicate the current direction of the vehicle in preparation for the subsequent calculation of the collision box. The current angle is obtained by calculating the angle between the last two points, again with the formula:

$$
\theta_{\text{current}} = \arctan\left(\frac{y_2 - y_1}{x_2 - x_1}\right)\quad\quad\quad (11)
$$

​	In this way, we can analyse angular variations in the vehicle trajectory, where excessive angular variations may indicate abnormal steering or sudden trajectory shifts.

#### Curvature Calculation

​	Curvature $$ \kappa $$ is used as a measure of the curvature of a trajectory and is calculated based on the geometric relationship of three consecutive points. Curvature is defined as the ratio of the product of the area of the triangle formed by these three points and the length of the side. The exact formula is:

$$
\kappa = \frac{2 \cdot |(x_2 - x_1)(y_3 - y_1) - (y_2 - y_1)(x_3 - x_1)|}{\sqrt{\left( (x_2 - x_1)^2 + (y_2 - y_1)^2 \right) \cdot \left( (x_3 - x_1)^2 + (y_3 - y_1)^2 \right)}}\quad\quad\quad (12)
$$

where:

1. $$ (x_1, y_1) $$, $$ (x_2, y_2) $$, $$ (x_3, y_3) $$ denote the coordinates of three consecutive trajectory points, respectively;

2. The numerator denotes twice the area of the triangle formed by these three points;

3. The denominator represents the product of the sum of the squares of the lengths of the sides adjacent to the triangle.

​	The formula calculates the area of the triangle using the vector fork product, normalised by the distance between successive points to ensure that the result matches the curvature of the curve.

​	In ARMS, the function *calculate_curvature* iterates over each of the three consecutive points in the nearest trajectory, calculates their curvature, and accumulates all the calculated curvature values. The average curvature of the entire trajectory is returned:

$$
\bar{\kappa} = \frac{\sum_{i=1}^{N} \kappa_i}{N}\quad\quad\quad (13)
$$

​	where $$ \kappa_i $$ is the curvature at each set of three points and $$ N $$ is the number of segments in the trajectory for which curvature can be calculated. A large curvature value means that the trajectory has a large bend or turn, while a curvature of 0 means that the trajectory is flat.

​	Angle analysis and curvature analysis have complementary roles in trajectory analysis. Angle analysis can quickly capture break out in the trajectory, which is suitable for real-time monitoring and detecting abnormal driving behaviours, while curvature analysis provides a global view of the entire trajectory pattern, which is able to detect smoother steering and curved motions over long periods of time. By combining the two analysis methods, we are able to gain a more comprehensive understanding of changes in vehicle trajectories, not only capturing sudden changes in direction, but also monitoring long-term trajectory curvature and steering behaviours. This is important in autonomous driving, risk assessment, and accident prevention.

#### Detection frame overlap calculation

​	During vehicle detection and tracking, the overlap of the detection frames is an important metric used to assess the proximity between two vehicles. The degree of overlap is usually calculated by the Intersection over Union (IoU, Intersection over Union) ratio, which is given by the formula:

$$
\text{IoU} = \frac{A_{\text{intersection}}}{A_{\text{union}}}\quad\quad\quad (14)
$$

where:

1. $$ A_{\text{intersection}} $$ is the area of the intersection region of the two detection frames;

2. $$ A_{\text{union}} $$ is the area of the concatenation region of the two detection frames.

However, in order to improve the robustness and cope with the situation where the gap between the detection target volumes is too large, a bidirectional overlap rate calculation method is used. This method calculates the ratio of the area of the intersection between the two detection frames to the area of each detection frame separately to more accurately reflect the collision relationship between the two vehicles. Its formula is:

$$
\text{MaxIoU} = \max\left( \frac{A_1 \cap A_2}{A_1}, \frac{A_1 \cap A_2}{A_2} \right)\quad\quad\quad (15)
$$

Among them:

1. $$ A_1 $$ and $$ A_2 $$ denote the area of the detection frames of the two vehicles, respectively;

2. $$ A_1 \cap A_2 $$ denotes the area of the intersection region of the two detection frames.

​	With the bidirectional overlap calculation, we can more accurately determine whether a vehicle is in a potential collision or not. rectangles are created in ARMS from the length, width, centroid and angle of the detection frames and these rectangles are used to calculate the overlap area to obtain the maximum overlap rate to represent the overlap of the detection frames of the two vehicles. If the distance between the vehicles exceeds the maximum detection frame diagonal, it is considered that there is no overlap between the two vehicles.

​	In the process of vehicle travelling, speed, overlap, angle change and curvature are four important physical quantities, which reflect the vehicle's motion state, relative positional relationship, change of trajectory direction and smoothness of travelling path, respectively. Abnormal fluctuations in these physical quantities may indicate an increase in risk, and thus the safety condition of a vehicle can be effectively assessed by analysing these four aspects. We can further integrate these different risk factors to generate a comprehensive risk score for each vehicle, which can be used to quantify the overall safety risk of the vehicle.

### Risk scoring function

​	In order to comprehensively assess the risk profile of a vehicle, ARMS takes into account factors such as speed [35], speed fluctuation, angle change [36], curvature [37], and detection frame overlap [38], etc. Inspired by the studies of S. G. Ritchie et al [39] and J. J. Lu et al [40], the authors designed the mapping function to provide a score for each of the factor separately, and finally the composite risk score was calculated by the maximum risk factor method. The following calculation method is a reasonable mapping method proposed by the authors through the analysis of this problem:

#### Speed Score

​	The speed score $$S_v$$ is given by the following formula:

$$
S_v = \min\left(\left(\frac{v}{1.3 \cdot v_0}\right)^4, 1\right) \cdot 10\quad\quad\quad (16)
$$

where:

1. $$v$$ indicates the current speed;

2. $$v_0$$ is the velocity threshold.

​	The speed score $$S_v$$ is based on the speed of the vehicle in relation to a preset threshold. The use of an exponential function to control the growth of the score enables a rapid increase in the score when the speed significantly exceeds the threshold. This approach is effective in reflecting the potentially high risk associated with excessive speed. The exponential function in the formula allows the score to grow slowly as the speed approaches the threshold and quickly reach the maximum score as the speed is well above the threshold.

#### Fluctuation Scoring

​	Volatility Scoring $$S_f$$ The exact formula is as follows:

$$
S_f = \min\left(\left(\frac{f}{\max(f_r \cdot v, 20)}\right)^2, 1\right) \cdot 10\quad\quad\quad (17)
$$

where:

1. $$f$$ denotes the magnitude of fluctuation;

2. $$f_r$$ is the fluctuation coefficient, which is used to regulate the relationship between fluctuation and velocity;

3. $$v$$ denotes the current velocity.

​	The fluctuation score $$S_f$$ incorporates the variability of vehicle speed, with a dynamic threshold influenced by this speed that enables the system to respond differently to fluctuations at varying velocities. At lower speeds, these fluctuations exert a more pronounced effect on safety, resulting in heightened sensitivity of the score. Additionally, the squaring function within the formula serves to amplify the risk associated with larger fluctuations.

#### Angle Scoring

​	Angle Scoring $$S_\theta$$ The exact formula is as follows:

$$
S_\theta = \min\left(\left(\frac{\theta}{\theta_0}\right)^2, 1\right) \cdot 10\quad\quad\quad (18)
$$

where:

1. $$\theta$$ is the change in angle of the neighbouring points of the trajectory;

2. $$\theta_0$$ is the preset angle threshold.

​	The angle score $$S_\theta$$ assesses the angle change of the vehicle trajectory. Excessive angle change may indicate that the vehicle is executing a sharp turn or lane change, increasing the risk of an accident. The incorporation of a squared function results the risk of angular change increase rapidly when the change of angle is significant.

#### Curvature Score

​	Curvature Scoring $$S_\kappa$$ The exact formula is as follows:

$$
S_\kappa = \min\left(\left(\frac{\kappa}{\max\left(\frac{v_0}{v} \cdot \kappa_0, 0.001\right)}\right)^2, 1\right) \cdot 10\quad\quad\quad (19)
$$

Among them:

1. $$\kappa$$ denotes the current curvature;

2. $$v$$ is the current velocity;

3. $$v_0$$ and $$\kappa_0$$ are the velocity and curvature thresholds, respectively.

​	The curvature score $$S_\kappa$$ computes the curvature change of the trajectory and takes into account the effect of vehicle speed. Higher speeds are less sensitive to curvature, so the threshold for scoring is dynamically adjusted. The contribution of curvature to risk can be efficiently assessed using the square function, especially at higher curvatures.

#### Overlap Degree Scoring

​	Overlap Degree Scoring $$S_o$$ The specific formula is as follows:

$$
S_o = \min\left(\left(\frac{o}{o_0}\right)^3, 1\right) \cdot 10\quad\quad\quad (20)
$$

where:

1. $$o$$ denotes the current overlap;

2. $$o_0$$ is the preset overlap degree threshold.

​	The overlap score $$S_o$$ is used to evaluate the overlap of the detection frame. Too much overlap usually means that two vehicles are close or colliding. The cubic function in the formula allows the overlap score to increase rapidly when the overlap is large, better reflecting the risk of overlap.

### Composite Risk Score Calculation

​	The Composite Risk Score $$S_{\text{total}}$$ is calculated using the Maximum Risk Factor method with the following formula:

$$
S_{\text{total}} = 0.5 \cdot S_{\text{avg}} + 0.5 \cdot S_{\text{max}}\quad\quad\quad (21)
$$

where:

1. $$S_{\text{avg}}$$ is the average of the velocity, fluctuation, angle, curvature and overlap scores;

2. $$S_{\text{max}}$$ is the maximum of these scores.

​	Reasons for using the maximum risk factor approach:

1. **Highlighting the Highest Risks**: The Maximum Risk Factor Method ensures that the most severe risk factors among all the scores have an impact on the composite score. This effectively reflects the one that is the most severe of the multiple risk factors, thus giving a more conservative safety assessment.

2. **Balanced scoring**: By considering both the average and maximum scores, it is possible to balance the smoothing of the overall risk with the suddenness of the worst-case scenario. The average score considers the overall performance of all risk factors, while the maximum score emphasises the most severe risk points.

3. **Enhanced robustness**: Combining the maximum and average scores helps to increase the robustness of the system to extreme situations and ensures that the composite scores remain accurate in high-risk situations.

​	In this way, the composite risk score can fully reflect the vehicle's risk performance in different aspects while emphasising the most critical risk factors.

### Accident detection logic

​	Accident detection is triggered when the composite score $$S_{\text{total}}$$ exceeds the threshold value $$S_{\text{threshold}}$$:

​	The system determines that there is a possibility of an accident occurring at this time and triggers the warning mechanism. The specific logic is as follows:

1. **Input:** Vehicle trajectory, speed, overlap, angle change and other parameters.
   
2. **Calculation:** 
   
   ​	(1) Calculate the score for each risk factor $$S_v, S_o, S_\theta, S_\kappa$$ separately.

   ​	(2) Composite risk score $$S_{\text{total}}$$ is calculated by weighted average.
   
3. **Decision making:**
   
   ​	(1) If $$S_{\text{total}} > S_{\text{threshold}}$$, it is determined that an incident has occurred and a warning signal is issued.

   ​	(2) Otherwise, the system continues to monitor and update the vehicle status and risk score in real time.
   
4. **Output:**
   
   ​	(1) Display the accident area and vehicle information, and record the exact time point and location of the accident.
   
   ​	(2) If the video save function is enabled, the frame and subsequent number of frames are saved to a local file.

### System Optimisation and Accident Detection Model Improvement

​	In order to improve the computational efficiency and detection accuracy, the following optimisations are also carried out in this study:

1. **Vectorisation calculation:** 

   To enhance calculation speed and efficiency, the computation of speed and angle changes is vectorized to eliminate unnecessary loops. This vectorized approach leverages NumPy's array operations and functions, thereby significantly accelerating the computational process.

2. **Trajectory smoothing process:** 

   To mitigate the effects of noise and jitter, inspired by the work of D. Comaniciu et al. [41], the authors implement a double smoothing process on vehicle trajectories; specifically, the original trajectories are smoothed first, followed by the smoothing of the transformed trajectories. The specific smoothing procedure employs a Gaussian filter function represented by the following equations.

    (1) **Gaussian filter function:**

   Gaussian filtering is a smoothing method based on the Gaussian distribution, and its filter function is:

   $$
G(x; \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{x^2}{2 \sigma^2}}\quad\quad\quad (22)
   $$
   
   where $$\sigma$$ represents the standard deviation of the Gaussian filter, which governs the degree of smoothing. A larger standard deviation results in a more pronounced smoothing effect

   (2) **Trajectory smoothing: **

   For the smoothing of the trajectory, Gaussian filter can be applied to the x-coordinate and y-coordinate of the trajectory respectively. Let the original trajectory be a collection of points $$(x_i, y_i)$$, and the smoothed trajectory $$(x‘_i, y’_i)$$ use the following formula:

   $$
x'_i = \frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^{\infty} x \cdot e^{-\frac{(x - x_i)^2}{2 \sigma^2}} \, dx\quad\quad\quad (23)
   $$
   
   $$
y'_i = \frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^{\infty} y \cdot e^{-\frac{(y - y_i)^2}{2 \sigma^2}} \, dy\quad\quad\quad (24)
   $$
   
   where $$x‘_i$$ and $$y’_i$$ are the smoothed x-coordinate and y-coordinate, respectively.

3. **Customised Risk Thresholds:** 

   Provides customizable risk threshold settings, enabling the sensitivity of incident detection to be dynamically adjusted based on varying scenarios.

4. **Anti-jitter processing:** 

   Avoid detecting false accident signals, especially when the vehicle is static state, by defining the static threshold.

5. **Specified frame processing:**

   In order to further optimise the use of computational resources and improve the processing speed, this study introduces an interval frames processing function. By skipping non-critical frames, unnecessary computations can be reduced while maintaining sensitivity to accident detection. With the specified frames processing strategy, the amount of computation can be significantly reduced and the overall efficiency of the system can be improved while ensuring the accuracy of detection.

## Experimental design and analysis of results

### Experimental setup

#### Datasets

​	A total of three datasets are used in the experiments: the vehicle detection dataset [18], the TAD accident dataset [19], and the UA-DETRAC [24]-[25]-[26] dataset, which are used to train the model to recognise vehicles and to test the system effectiveness.

​	The vehicle detection dataset is a road traffic image dataset acquired from an unmanned aerial vehicle (UAV) and contains two categories, car and motorcycle. The dataset consists of 15,070 images in PNG format accompanied by an equal number of files with txt extension containing descriptions of the recognised elements in each image. In total, there are 30,140 files containing images and descriptions. The images were taken on urban and intercity roads at six different locations; images on motorways were excluded. From these, the authors selected data suitable for use in this task for experimentation: after random disruption and formatting, the dataset was modified to a format that could be trained by yolo. There are 6151 images in the training set and 2018 images in the validation set.

​	The TAD accident dataset is a large-scale open-source traffic accident video dataset that provides a richer variety of accident types and scenarios than other traffic accident datasets such as UCF-Crime [20], CADP [21] and DAD [22]. The video data comes from different surveillance cameras, including those collected from traffic video analytics platforms and mainstream video sharing websites (e.g., Weibo). It contains 333 videos with resolutions ranging from (862,530) to (1432,990), covering 261 samples containing traffic accidents.

​	The UA-DETRAC dataset is a challenging real-world multi-target detection and multi-target tracking benchmark test. The dataset consists of 10 hours of video shot with a Cannon EOS 550D camera at 24 different locations in Beijing and Tianjin, China. The videos were recorded at 25 frames per second (fps) with a resolution of 960 x 540 pixels.

#### Experimental Environment

​	All experiments in this paper were conducted on a computer configured with the following system and equipment parameters: Windows 11, version 23H2 operating system with an Intel(R) Core(TM)i5-13400@4.60GHz CPU and an NVIDIA GeForce RTX 4070Ti (12G) GPU, and 64G RAM. The code for data processing and algorithm implementation was written in python 3.11.5 and implemented in Pytorch 2.0.1 deep learning framework. pycharm developed by Jetbrains was used for the IDE.

​	It is worth mentionable that, in order to ensure the accuracy of the algorithm and to consider the dependence of the algorithm on the vehicle detection model, a pre-model of yolov10x was used for training, with a total of 200 epochs, a batch size of 8, a number of workers of 2, the optimiser was chosen to be SGD, and the rest of the configurations were made by using the default configuration of yolov10x. Configuration.

### Vehicle Detection Model Evaluation

The training took a total of 11.65h, and the metrics changed during training as follows:

![image-20241028113916841](C:/Users/kitag/AppData/Roaming/Typora/typora-user-images/image-20241028113916841.png)

​	*Figure 5 shows the change of the loss function during training and validation, which mainly contains three parts of loss*

1. **box_loss**: The bounding box-loss reflects the deviation between the model-predicted bounding box and the real box. This loss decreases as training proceeds, indicating a gradual improvement in the model's prediction of the bounding box.
2. **cls_loss**: The classification loss reflects the model's recognition error on the target category. The curve shows that this loss gradually decreases with training, indicating that the classification accuracy is improving.
3. **dfl_loss**: The distribution focal loss (DFL) reflects the loss of fine tuning to the bounding box. This loss decreases rapidly in the early stages and then levels off.

​	Comparing the losses on the validation set to the test set, the loss on the validation set decreases in a similar trend to the training set, but the validation loss does not decrease as fast as the training loss, indicating that the model is slightly less effective on the validation set than on the training set.

​	Overall, all losses gradually stabilise in the later stages of training, indicating that the model gradually converges.

Figure 6

![Metrics](https://s2.loli.net/2024/09/16/G97Ozvrfd3QU8kW.png)

​	Figure 6 shows the changes of various evaluation metrics during the model training process, covering the following four metrics:

1. **Precision**: The precision initially measured approximately 0.85, exhibiting fluctuations before achieving stability and oscillating within a range close to 1.0, thereby indicating that the model demonstrates a high level of accuracy.
2. **Recall**: The recall initially measured approximately 0.75, and gradually improves and approaches 1.0 with increased training, indicating that the target acquisition capability is gradually increasing.
3. **mAP50**: The average precision while IoU is 0.5. The curve begins to rise sharply from around 0.65, subsequently stabilizing and ultimately oscillating around 0.9.
4. **mAP50-95**: The average precision (from 0.5 to 0.95) is at multiple IoU thresholds. The rising curve of this value is slower than that of mAP50, and eventually stabilizes around 0.8, indicating that the model also performs well in terms of accuracy under different IoU thresholds.

Figure 7

![Confusion Matrix](https://s2.loli.net/2024/09/16/hA6PRiQgUEMw5tG.png)

​	Figure 7 shows the confusion matrix, which demonstrates how the model's prediction results match the actual labels for the three types of targets (**car**, **motorbike**, **background**) to evaluate the performance of the classification model.

​	The model performed very well for `car`, correctly classifying 21867 samples, with only a small number of misclassifications to `background` (398), and no samples were misclassified as `motorbike`.

​	The model also performed well in classifying `motorbike`, with 1166 samples correctly classified, but 29 misclassified as `background`, showing that the `motorbike` category can be difficult to distinguish in some cases.

​	The model performs weakly when classifying `background`. It misclassified not only 92 `background` samples as `car` but also 171 `background` samples as `motorbike`.

​	In summary, the model complishes a gradual reduction in loss throughout the training process, with evaluation metrics stabilizing and achieving satisfactory convergence. Both precision and recall approach 1.0, indicating that the model effectively detects positive samples while maintaining low false alarm rates. Furthermore, the increase in average precision suggests enhanced detection capabilities across various IoU thresholds, particularly highlighting its exceptional performance in accuracy and bounding box localization.

### Evaluation of system operation results

Figure 8

![image-20240917124342001](https://s2.loli.net/2024/09/21/pAaUKTj5ChPM3lg.png)

​	Figure 8 shows the results of the system

​	In this study, we randomly selected 25 video segments from both the TAD dataset and the UA-DETRAC dataset to evaluate system performance and assess the model's recognition capabilities. Each selected video segment is uniformly 15 seconds in duration, with a frame rate of 25 frames per second and a resolution of 960 × 540 pixels, encompassing a diverse range of temporal weather conditions and scenarios. The video segments from the TAD dataset include key frames depicting accidents, aimed at evaluating the system's sensitivity to such events, while those provided by the UA-DETRAC dataset represent normal traffic flow and are utilized for assessing the false alarm rate of the system.

​	To facilitate a quantitative analysis of system performance and mitigate the influence of subjective factors on evaluation outcomes, this study adopts the methodology proposed by X. Huang et al. in reference [27]. Specifically, an event segment is defined as a time window encompassing 1 second before and 1 second after the incident, totaling 2 seconds. This definition serves as the smallest temporal unit for determining whether an incident has occurred. Consequently, if the system identifies an incident within this 2-second window, it is deemed to have detected it accurately. Conversely, if the system reports an incident outside this time window when no actual incident occurs, it is classified as a false alarm. This approach enables a more accurate assessment of the system's effectiveness. It is important to note that for each video, the conversion parameters and associated thresholds are readjusted to tailor detection to the specific context.

​	Utilizing the aforementioned evaluation approach, the resulting confusion matrix is presented in Fig. 9, along with the corresponding performance metrics, including accuracy, recall, precision, and F1 score:

Figure 9

![ARMS Confusion Matrix](https://s2.loli.net/2024/09/21/9Iucv7Rq1UbMgfe.png)

**True Positive (TP)**: The volume of data that the system accurately predicts as an incident.

**False Positive (FP)**: The volume of non-accident data that the system erroneously predicts as an accident.

**True Negative (TN)**: The volume of data that the system accurately predicts as non-accidental.

**False Negative (FN)**: The volume of accident data that the system erroneously predicted as non-accident.

**Accuracy**: 

$$
Accuracy=\frac {TP+TN}{TP+TN+FP+FN}=\frac {23+21}{23+21+4+2}=0.88\quad\quad\quad (25)
$$
Accuracy reflects that the system correctly identifies 88% of all predictions.

**Recall**:

$$
Recall=\frac {TP}{TP+FN}=\frac {23}{23+2}=0.92\quad\quad\quad (26)
$$
The recall rate, also referred to as the true positive rate, quantifies the percentage of incidents accurately identified by the system, which stands at 92%.

**Precision**: 
$$
Precision=\frac {TP}{TP+FP}=\frac {23}{23+4}≈0.852\quad\quad\quad (27)
$$
Precision quantifies the proportion of data predicted by the system as accidents that are actual accidents, which is 85.2%.

**F1 Score (F1 Score)**: 

$$
F1 Score=2×\frac{Precision×Recall}{Precision+Recall}=2×\frac{0.852×0.92}{0.852+0.92}≈0.884\quad\quad\quad (28)
$$
​	The F1 score represents the harmonic mean of precision and recall, effectively balancing the two metrics. In this study, the F1 score is 88.4%, indicating that the system achieves a commendable equilibrium between precision and recall.

​	The results demonstrate that ARMS exhibits robust performance in the accident detection task. The accuracy, recall, and precision rates collectively indicate that the system effectively identifies both accident and non-accident scenarios. Notably, the F1 score—integrating precision and recall—serves as a more comprehensive metric for evaluating the system's performance. Elevated values of the F1 score suggest that the system achieves an optimal balance between minimizing false alarms (enhancing precision) and reducing missed detections (enhancing recall). 

​	Furthermore, the authors investigate the performance of various model architectures under specified frame processing and non-specified frame processing conditions. Specified frame processing refers to the system's operation on one out of every 15 frames, while non-specified frame processing entails that the system processes all frames, thereby offering a more realistic representation of the system's operational speed limitations.

| Using Model Architecture           | Designated Frame Processing | FPS   |
| ---------------------------------- | --------------------------- | ----- |
| **v10x (used on this system)**[11] | √                           | 15.01 |
| v10n[11]                           | √                           | 15.03 |
| v10x[11]                           |                             | 21.05 |
| v10n[11]                           |                             | 24.99 |
| v8x[28]                            | √                           | 15.00 |
| v8n[28]                            | √                           | 15.01 |
| v8x[28]                            |                             | 20.74 |
| v8n[28]                            |                             | 24.97 |
| v5x[29]                            | √                           | 14.98 |
| v5n[29]                            | √                           | 15.02 |
| v5x[29]                            |                             | 21.23 |
| v5n[29]                            |                             | 25.00 |

​	*Table 2 shows the average results obtained from ten videos of 10 minutes in length cropped from the UA-DETRAC dataset and the TAD dataset, with a default frame rate of 25 fps for the videos. The pixel size is uniformly 960 × 540.*

​	The experimental results indicate that the N model achieves a processing speed of 25 fps without employing specified frame processing, aligning with the default frame rate of the surveillance video. This demonstrates that the system effectively utilizes video data for efficient real-time surveillance. In contrast, the X model attains a processing speed of approximately 21 fps under identical conditions, which, while slightly lower than that of the N model, still satisfies basic accident detection sampling requirements.

​	Furthermore, the authors observe that the operational efficiency of the code is proportional to the square of the number of vehicles, particularly during accident detection sessions. This phenomenon arises from the necessity for each vehicle to be compared with all other vehicles; consequently, as the number of vehicles increases, the computational demands escalate exponentially. As a result, the system can operate at high speeds in scenarios such as highways or rural roads where traffic is sparse. In contrast, on congested roadways like city centers where traffic density is high, system performance may be constrained. Nevertheless, it remains capable of ensuring a minimum processing speed of at least 20 frames per second for real-time monitoring.

​	Although specified frame processing reduces the system's frames per second (FPS), this strategy offers significant advantages for handling UHD and high frame rate camera data. By minimizing computational resource consumption, specified frame processing aids in sustaining stable system performance during real-time operations, particularly in resource-constrained environments. Consequently, designated frame processing emerges as an effective strategy to optimize the efficiency of resource utilization while preserving surveillance quality.

### Comparison

​	To illustrate the superiority of this algorithm, the authors compared its performance metrics with those obtained from models within the YOLO family for accident target detection, which were trained using the accident dataset (Table 3), employing identical evaluation criteria:

Table 3

| Algorithm | Accuracy | Recall | Precision | F1 Score |
| --------- | -------- | ------ | --------- | -------- |
| ARMS      | 0.88     | 0.92   | 0.85      | 0.88     |
| YoloV10x  | 0.76     | 0.88   | 0.71      | 0.79     |
| YoloV8x   | 0.74     | 0.80   | 0.71      | 0.76     |
| YoloV5x   | 0.68     | 0.76   | 0.66      | 0.71     |

​	The results indicate that the ARMS algorithm achieves superior performance across all four metrics: accuracy, recall, precision, and F1 score. Notably, there is a substantial improvement in addressing false alarms (i.e., misclassifying a non-accident as an accident). This demonstrates that the proposed system, based on YOLOv10 and Bot-SORT algorithms, significantly enhances performance for the Vision-TAD problem, particularly concerning false alarm reduction and validates the efficacy of both the vision transformation method and risk assessment approach.

​	Additionally, the authors provide a brief comparison of this system with representative Vision-TAD works from recent years (Table 4). This comparison serves merely as a reference, given that the evaluation criteria are not consistent across different studies:

Table 4

| Years | Algorithm                                                    | Models                   | Recall     | Precision | F1 Score  |
| ----- | ------------------------------------------------------------ | ------------------------ | ---------- | --------- | --------- |
| 2024  | ARMS                                                         | Yolo+Bot-Sort            | 0.92       | 0.85      | 0.88*     |
| 2021  | Vehicular Trajectory Classification and Traffic Anomaly Detection[32] | CNN+VAE                  | 0.93*      | 0.82      | 0.87      |
| 2020  | Two-Stream Convolutional Networks[30]                        | CNN                      | 0.83       | 0.89*     | 0.86      |
| 2020  | GBLSTM2L+A SILSTM Networks[31]                               | Siamese Interaction LSTM | 0.755(avg) | 0.50(avg) | 0.60(avg) |

​	The Recall of the ARMS algorithm ranks among the highest across various algorithms, demonstrating its effectiveness in detecting most anomalous vehicle trajectories. In contrast, the Recall of the 2020 GBLSTM2L+A SILSTM Networks is only 0.755, highlighting a significant disparity; although the Precision of ARMS is slightly lower than that of Two-Stream Convolutional Networks at 0.89, it still maintains an admirable level of accuracy. Conversely, the Precision of GBLSTM2L+A SILSTM Networks is merely 0.50, indicating a high false alarm rate in its detection results. The F1 Score for ARMS is the highest among all evaluated algorithms, emphasizing its superiority in balancing Recall and Precision. Notably, compared to earlier algorithms, ARMS exhibits a distinct advantage in overall performance.

​	In summary, this experiment validates the efficient performance of the ARMS system in terms of accuracy, recall, and precision by evaluating both model detection results and operational outcomes. Furthermore, it demonstrates commendable operational speed under both specified and unspecified frame processing conditions.

## Discussions

### System Optimisation

​	Regarding system performance optimization, while the current system has successfully achieved fundamental functions such as vehicle detection, trajectory tracking, and risk assessment, a bottleneck in efficiency remains when processing multiple vehicles simultaneously. To address this challenge, optimization efforts can focus on both algorithmic and data structure levels. By incorporating spatial trees (e.g., KD tree or quadtree), the search and matching of targets can be accelerated, thereby reducing processing time—particularly in scenarios with a high volume of vehicles. Additionally, limiting the number of vehicle identifications per processing cycle represents another effective strategy; prioritizing the identification of high-risk vehicles ensures that key targets receive adequate attention while optimizing resource utilization.

​	Simultaneously, errors in the mapping process can significantly impact the system's analysis of trajectories and risk assessment. Therefore, enhancing the matching accuracy between trajectories and actual coordinates is essential. Mapping inaccuracies can be mitigated through more precise geometric transformation algorithms and multi-point calibration to ensure the integrity of trajectory data.

### Algorithm improvement

​	The detection algorithm is pivotal for enhancing system performance. While the currently employed YOLOv10 algorithm has achieved a reasonable balance between speed and accuracy, issues such as false detections and missed detections may still arise in complex scenarios. For instance, the system's target detection and tracking algorithm exhibits challenges like flickering and mutation of the detection frame during practical applications, particularly when vehicles are in rapid motion or partially obscured. This flickering results in an unstable tracking process, adversely affecting risk assessment accuracy. To address this issue, a long time-series information fusion method can be implemented to smooth positional changes of the detection frame using historical data, thereby reducing flickering phenomena and improving stable tracking performance.

​	Regarding trajectory tracking and risk assessment, trajectory imbalance presents a significant challenge. When vehicles are positioned too far from the camera, reliance on image information leads to decreased accuracy in trajectory extraction; consequently, the mapped trajectory diverges significantly from that obtained at closer ranges. Current smoothing and tracking algorithms may inadequately address this imbalance. In future work, dynamically adjusted smoothing parameters and multi-scale trajectory processing techniques could be implemented to ensure appropriate handling of trajectories with varying complexities, thereby enhancing the reliability and accuracy of risk assessments.

​	Enhancing risk assessment methods is crucial for the scientific advancement of the system. While existing risk assessment techniques are effective, there is a need for a more systematic risk indicator framework to improve both interpretability and accuracy of the assessment results. Additionally, numerous parameters must be adjusted for various scenarios to ensure adaptability in specific contexts. In future developments, we aim to employ artificial intelligence methodologies to map physical feature information onto risk scores, thereby constructing a more comprehensive risk assessment model. Concerning accident occurrence assessments, the judgment logic based on a single threshold lacks rigor in complex scenarios; thus, we will explore dynamic threshold adjustments tailored to different situations moving forward.

### Future work direction

​	In the future, the system will undergo extensive testing and validation to ensure its stability and reliability across a variety of complex traffic scenarios. Through this testing, various modules—including target detection, trajectory tracking, and risk assessment—can be continuously optimized to minimize misdetections and omissions while further enhancing the system's real-time performance and robustness.

​	Regarding the expansion of system functionality, future developments will support more complex traffic scenarios, including multi-lane configurations and the detection and recognition of large vehicles under specific conditions such as nighttime and adverse weather. Additionally, exploring the feasibility of cross-scene applications represents a significant research direction, aiming to apply the system in diverse environments such as highways, urban roads, and even parking facilities to assess its adaptability and scalability.

​	Simultaneously, reducing the complexity of the system code will be a primary focus of future work. Streamlining the code structure and enhancing modularity and maintainability will not only facilitate system expansion and optimization but also improve code execution efficiency. By refactoring complex algorithms and functional modules, eliminating redundant code, and optimizing data flow and resource management, the computational load on the system can be significantly reduced, thereby enhancing overall performance.

​	The performance optimization of the system will also encompass the further application of technologies such as parallel computing and hardware acceleration, maximizing the capabilities of multi-core processors and graphics processing units to ensure that the system maintains a high level of efficiency and real-time performance when handling large-scale data. Additionally, optimizing memory management and resource allocation can reduce memory consumption while enhancing both stability and processing speed.

​	Ultimately, the system will undergo further validation in terms of scientific rigor and accuracy, particularly regarding the enhancement of risk assessment methods. By integrating expert knowledge from the field of traffic safety with actual accident data, the algorithms and models will be continuously optimized to ensure that the system can deliver more accurate and efficient solutions for traffic safety.

## Conclusion

### Summary of research results

​	This research presents the design and implementation of a deep learning-based traffic accident detection and risk assessment system, which effectively integrates various techniques such as target detection, trajectory tracking, trajectory smoothing, and risk assessment to facilitate the automatic identification and warning of traffic accidents. Experimental validation demonstrates that the system exhibits robust stability and real-time performance across multiple complex traffic scenarios. Notably, in terms of risk assessment, the system is capable of effectively analyzing vehicle trajectories, predicting potential risks in advance, and providing timely warnings. By optimizing the target detection and trajectory smoothing algorithms, significant improvements in system performance are achieved when managing multiple vehicles, thereby further enhancing its real-time processing capabilities.

​	A significant innovation of this system is the implementation of a viewpoint conversion method, which effectively mitigates detection and tracking errors across varying perspectives. Additionally, the scientific risk assessment methodology reduces the false alarm rate, ensuring stable operation of the system in diverse environments and complex scenarios.

### Research Contribution

​	This research makes significant contributions to the field of traffic safety. The developed intelligent accident detection system minimizes reliance on manual monitoring and offers an efficient, automated solution for accident identification and warning. By analyzing traffic flow and vehicle trajectories in real time, the system enhances road safety by providing timely alerts prior to accidents occurring. The findings of this research provide novel technical support for traffic safety management, particularly in reducing the false alarm rate associated with accident detection.

​	Furthermore, this study illustrates the potential of deep learning techniques in enhancing traffic safety and serves as a valuable reference for advancing the intelligence of traffic safety systems in the future. By optimizing the YOLOv10 target detection algorithm and trajectory smoothing algorithm, this research addresses the trade-off between real-time performance and accuracy, demonstrating the effectiveness and robustness of deep learning technology in complex scenarios.

​	This study also innovatively proposes perspective transformation and scientific risk assessment methods, which reduces the false alarm rate while enhancing detection accuracy and system stability.. This series of findings establishes a significant practical foundation for future research and applications of deep learning in the field of traffic safety.

## Reference

[1] World Health Organization. Global status report on road safety 2023. https://iris.who.int/bitstream/handle/10665/375016/9789240086517-eng.pdf?sequence=1. 

[2] Smith, B., & Jones, A. (2020). The role of CCTV in traffic accident detection: A review. Journal of Transportation Safety, 12(3), 234-245. 

[3] Lee, J., & Kim, J. (2021). Performance evaluation of manual traffic monitoring systems in urban areas. International Journal of Traffic and Transportation Engineering, 9(2), 112-120. 

[4] K. Jain and S. Kaushal, "A Comparative Study of Machine Learning and Deep Learning Techniques for Sentiment Analysis," *2018 7th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO)*, Noida, India, 2018, pp. 483-487, doi: 10.1109/ICRITO.2018.8748793.

[5] J. Fang, J. Qiao, J. Xue and Z. Li, "Vision-Based Traffic Accident Detection and Anticipation: A Survey," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 4, pp. 1983-1999, April 2024, doi: 10.1109/TCSVT.2023.3307655.

[6] Bemposta Rosende, S.; Ghisler, S.; Fernández-Andrés, J.; Sánchez-Soriano, J. Dataset: Traffic Images Captured from UAVs for Use in Training Machine Vision Algorithms for Traffic Management. *Data* 2022, *7*, 53. https://doi.org/10.3390/data7050053

[7] A. Mousavian, D. Anguelov, J. Flynn and J. Košecká, "3D Bounding Box Estimation Using Deep Learning and Geometry," *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, 2017, pp. 5632-5640, doi: 10.1109/CVPR.2017.597.

[8] WANG Chen, ZHOU Wei, YAN Jun-yi, GONG Yao-hui. Improved Two-stream Network for Vision-based Traffic Accident Detection[J]. *China Journal of Highway and Transport*, 2023, 36(5): 185-196 https://doi.org/10.19721/j.cnki.1001-7372.2023.05.016

[9] LI Hai-tao, LI Zhi-hui, WANG Xin, PAN Zhao-tian, QU Zhao-wei. Real-time Automatic Method of Detecting Traffic Incidents Based on Temporal Convolutional Autoencoder Network[J]. *China Journal of Highway and Transport*, 2022, 35(6): 265-276 https://doi.org/10.19721/j.cnki.1001-7372.2022.06.022

[10] LYU Pu, BAI Qiang, CHEN Lin. A Model for Predicting the Severity of Accidents on Mountainous Expressways Based on Deep Inverted Residuals and Attention Mechanisms[J]. *China Journal of Highway and Transport*, 2021, 34(6): 205-213 https://doi.org/10.19721/j.cnki.1001-7372.2021.06.020

[11] Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024). *YOLOv10: Real-time end-to-end object detection*. arXiv. https://arxiv.org/abs/2405.14458

[12] Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust associations multi-pedestrian tracking*. arXiv. https://arxiv.org/abs/2206.14651

[13] Ministry of Housing and Urban-Rural Development of the People's Republic of China. Code for design of urban road engineering: CJJ 37-2012 [S]. 2016 Edition. Beijing: China Architecture & Building Press, 2016. 

[14] Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. "BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View." arXiv preprint arXiv:2112.11790 (2022). Available at: https://arxiv.org/abs/2112.11790.

[15] Yunpeng Zhang, Jiwen Lu, and Jie Zhou. "Objects Are Different: Flexible Monocular 3D Object Detection." In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 3289-3298, June 2021.

[16] H. Caesar *et al*., "nuScenes: A Multimodal Dataset for Autonomous Driving," *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, 2020, pp. 11618-11628, doi: 10.1109/CVPR42600.2020.01164.

[17] Ming-Fang Chang, John W. Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays. "Argoverse: 3D Tracking and Forecasting with Rich Maps." In *Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.\

[18] Bemposta Rosende, S.; Ghisler, S.; Fernández-Andrés, J.; Sánchez-Soriano, J. Dataset: Traffic Images Captured from UAVs for Use in Training Machine Vision Algorithms for Traffic Management. Data 2022, 7, 53.

[19] Yajun Xu, Chuwen Huang, Yibing Nan, and Shiguo Lian. "TAD: A Large-Scale Benchmark for Traffic Accidents Detection from Video Surveillance." arXiv preprint arXiv:2209.12386*, 2022. Available at: https://arxiv.org/abs/2209.12386.

[20] W. Sultani, C. Chen and M. Shah, "Real-World Anomaly Detection in Surveillance Videos," *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, Salt Lake City, UT, USA, 2018, pp. 6479-6488, doi: 10.1109/CVPR.2018.00678.

[21] Ankit Shah*, Jean Baptiste Lamare*, Tuan Nyugen Anh*, Alexander Hauptmann “CADP: A Novel Dataset for CCTV Traffic Camera based Accident Analysis” International Workshop on Traffic and Street Surveillance for Safety and Security, Nov 2018.

[22] Okan Kopuklu, Jiapeng Zheng, Hang Xu, and Gerhard Rigoll. "Driver anomaly detection: A dataset and contrastive learning approach." In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 91-100, 2021.

[23] Ministry of Public Security of the People's Republic of China. GA/T 832—2014, Technical specifications for image forensics of road traffic offences [S]. Beijing: Ministry of Public Security of the People's Republic of China, 2014.

[24] Longyin Wen, Dawei Du, Zhaowei Cai, Zhen Lei, Ming-Ching Chang, Honggang Qi, Jongwoo Lim, Ming-Hsuan Yang, and Siwei Lyu. "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking." *Computer Vision and Image Understanding* (2020).

[25] Siwei Lyu, Ming-Ching Chang, Dawei Du, Wenbo Li, Yi Wei, Marco Del Coco, Pierluigi Carcagnì, Arne Schumann, Bharti Munjal, Doo-Hyun Choi, et al. "UA-DETRAC 2018: Report of AVSS2018 & IWT4S Challenge on Advanced Traffic Monitoring." In *2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*, 1-6. IEEE, 2018.

[26] Siwei Lyu, Ming-Ching Chang, Dawei Du, Longyin Wen, Honggang Qi, Yuezun Li, Yi Wei, Lipeng Ke, Tao Hu, Marco Del Coco, et al. "UA-DETRAC 2017: Report of AVSS2017 & IWT4S Challenge on Advanced Traffic Monitoring." In *2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*, 1-7. IEEE, 2017.

[27] X. Huang, P. He, A. Rangarajan, and S. Ranka, “Intelligent inter section: Two-stream convolutional networks for real-time near accident detection in traffic video,” ACM Trans. Spatial Algorithms Syst., vol. 6, no. 2, pp. 10:1–10:28, 2020.

[28] R. Varghese and S. M., "YOLOv8: A Novel Object Detection Algorithm with Enhanced Performance and Robustness," *2024 International Conference on Advances in Data Engineering and Intelligent Computing Systems (ADICS)*, Chennai, India, 2024, pp. 1-6, doi: 10.1109/ADICS58448.2024.10533619.

[29] Yolov5 Contributors. (2024). GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite [Computer software]. GitHub. https://github.com/ultralytics/yolov5

[30]  X. Huang, P. He, A. Rangarajan, and S. Ranka, “Intelligent inter section: Two-stream convolutional networks for real-time near accident detection in traffic video,” ACM Trans. Spatial Algorithms Syst., vol. 6, no. 2, pp. 10:1–10:28, 2020.

[31] D. Roy, T. Ishizaka, C. K. Mohan and A. Fukuda, "Detection of Collision-Prone Vehicle Behavior at Intersections Using Siamese Interaction LSTM," in *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 4, pp. 3137-3147, April 2022, doi: 10.1109/TITS.2020.3031984.

[32] K. Kumaran Santhosh, D. P. Dogra, P. P. Roy and A. Mitra, "Vehicular Trajectory Classification and Traffic Anomaly Detection in Videos Using a Hybrid CNN-VAE Architecture," in *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 11891-11902, Aug. 2022, doi: 10.1109/TITS.2021.3108504.

[33] S. B. Kang, J. Y. Choi, and S. I. Yoo, "A Survey on Projection Transformation Techniques for Computer Vision Applications," IEEE Access, vol. 8, pp. 52885-52895, 2020.

[34] M. Enzweiler, U. Franke, and I. M. Herrmann, "A Multi-Feature Vehicle Detection Approach for Robust and Reliable Traffic Analysis," in Proc. IEEE Intelligent Vehicles Symposium, 2011, pp. 1-6.

[35] R. T. Huang and M. Q.-H. Meng, "Vehicle Speed Estimation Using Computer Vision Techniques: A Review," in Proc. IEEE International Conference on Vehicular Electronics and Safety, 2008, pp. 1-5.

[36] A. M. Wilson and A. F. Yilmaz, "Tracking and analyzing multiple objects in traffic video," in Proc. IEEE Conference on Advanced Video and Signal Based Surveillance, 2007, pp. 4-9.

[37] J. J. Koenderink and W. J. van Doorn, "Generic neighborhood operators," Journal of Mathematical Imaging and Vision, vol. 4, no. 1, pp. 35-62, 1994.

[38] N. Dalal, B. Triggs, and C. Schmid, "Human detection using oriented histograms of flow," in Proc. European Conference on Computer Vision, 2006, pp. 428-441.

[39] S. G. Ritchie, "A framework for assessing the safety risk of motor vehicle crashes," Accident Analysis & Prevention, vol. 42, no. 5, pp. 1318-1326, 2010.

[40] J. J. Lu, C. Y. Juang, and C. C. Lin, "A risk assessment model for highway work zones," Accident Analysis & Prevention, vol. 39, no. 3, pp. 590-598, 2007.

[41] D. Comaniciu, V. Ramesh and P. Meer, "Real-time tracking of non-rigid objects using mean shift," *Proceedings IEEE Conference on Computer Vision and Pattern Recognition. CVPR 2000 (Cat. No.PR00662)*, Hilton Head, SC, USA, 2000, pp. 142-149 vol.2, doi: 10.1109/CVPR.2000.854761.

## Appendix

Experimental code details are available at: https://github.com/Kitagawayyds/Traffic-accident-prediction
