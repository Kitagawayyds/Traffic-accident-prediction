import os
import glob
import shutil
import numpy as np

dataset = 'sec3'

def get_filenames(folder):
    filenames = set()

    for path in glob.glob(os.path.join(folder, '*.png')):
        filename = os.path.split(path)[-1]
        filenames.add(filename)

    return filenames

section = get_filenames(f'C:\\Users\kitag\Desktop\dataset\{dataset}')

np_section = np.array(list(section))

np.random.seed(69)
np.random.shuffle(np_section)

total_images = len(np_section)
train_size = int(0.6 * total_images)
val_size = int(0.2 * total_images)
test_size = total_images - train_size - val_size

print(f'Total images: {total_images}, Train size: {train_size}, Val size: {val_size}, Test size: {test_size}')

def split_dataset(label, image_names, train_size, val_size, test_size):
    for i, image_name in enumerate(image_names):
        label_name = image_name.replace('.png', '.txt')

        if i < train_size:
            split = 'train'
        elif i < train_size + val_size:
            split = 'val'
        else:
            split = 'test'

        source_image_path = f'C:\\Users\kitag\Desktop\dataset\{label}\\{image_name}'
        source_label_path = f'C:\\Users\kitag\Desktop\dataset\{label}\\{label_name}'

        target_image_folder = f'dataset2/images/{split}'
        target_label_folder = f'dataset2/labels/{split}'

        os.makedirs(target_image_folder, exist_ok=True)
        os.makedirs(target_label_folder, exist_ok=True)

        shutil.copy(source_image_path, target_image_folder)
        shutil.copy(source_label_path, target_label_folder)

split_dataset(dataset, np_section, train_size, val_size, test_size)
