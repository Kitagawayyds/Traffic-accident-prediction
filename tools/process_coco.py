import os
import shutil

# 定义允许的类别ID
allowed_classes = {2, 3, 5, 7}
'''
  2: car
  3: motorcycle
  5: bus
  7: truck
'''
# 定义新的类别ID映射
class_id_mapping = {2: 0, 3: 1, 5: 2, 7: 3}

# 数据集的路径
dataset_path = 'C:\\Users\kitag\PycharmProjects\datasets\coco'  # 替换为你的数据集路径
train_images_path = os.path.join(dataset_path, 'images', 'train2017')
train_labels_path = os.path.join(dataset_path, 'labels', 'train2017')
val_images_path = os.path.join(dataset_path, 'images', 'val2017')
val_labels_path = os.path.join(dataset_path, 'labels', 'val2017')

# 创建新的训练和验证文件夹
new_train_images_path = os.path.join(dataset_path, 'images', 'train')
new_train_labels_path = os.path.join(dataset_path, 'labels', 'train')
new_val_images_path = os.path.join(dataset_path, 'images', 'val')
new_val_labels_path = os.path.join(dataset_path, 'labels', 'val')

os.makedirs(new_train_images_path, exist_ok=True)
os.makedirs(new_train_labels_path, exist_ok=True)
os.makedirs(new_val_images_path, exist_ok=True)
os.makedirs(new_val_labels_path, exist_ok=True)

# 函数用于处理训练或验证数据
def process_data(images_path, labels_path, new_images_path, new_labels_path):
    for img_name in os.listdir(images_path):
        if img_name.endswith('.jpg'):  # 假设图片格式为jpg，根据实际情况修改
            label_name = img_name.replace('.jpg', '.txt')
            label_path = os.path.join(labels_path, label_name)
            new_label_path = os.path.join(new_labels_path, label_name)

            if not os.path.exists(label_path):
                print(f"警告: {img_name} 没有对应的标签文件，将跳过此图片。")
                continue

            # 读取标签文件
            with open(label_path, 'r') as file:
                lines = file.readlines()

            # 过滤和更新标签
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in allowed_classes:
                    new_class_id = class_id_mapping.get(class_id, class_id)  # 更新类别ID
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)

            # 如果有保留的标签，则复制图片和更新标签
            if new_lines:
                shutil.copy(os.path.join(images_path, img_name), os.path.join(new_images_path, img_name))
                with open(new_label_path, 'w') as new_file:
                    new_file.writelines(new_lines)

# 处理训练数据
process_data(train_images_path, train_labels_path, new_train_images_path, new_train_labels_path)
# 处理验证数据
process_data(val_images_path, val_labels_path, new_val_images_path, new_val_labels_path)

print("处理完成！")