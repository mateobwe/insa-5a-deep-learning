import os
import random
import shutil

train_directory = "dataset/train"
val_directory = "dataset/val"
data_set_percent_size = 0.8

os.makedirs(train_directory + "/images", exist_ok=True)
os.makedirs(train_directory + "/labels", exist_ok=True)
os.makedirs(val_directory + "/images", exist_ok=True)
os.makedirs(val_directory + "/labels", exist_ok=True)

for video_number in [1, 2, 3, 4, 5]:
    txt_directory = f"dataset/labelled/video{video_number}/obj_train_data"
    jpg_directory = f"data/frames/img{video_number}"

    txt_files = {f[:-4] for f in os.listdir(txt_directory) if f.endswith('.txt')}
    jpg_files = {f[:-4] for f in os.listdir(jpg_directory) if f.endswith('.jpg')}

    # Only keep basenames that exist in BOTH sets
    common = list(txt_files & jpg_files)

    random.shuffle(common)
    train_count = int(len(common) * data_set_percent_size)

    train_basenames = set(common[:train_count])
    val_basenames = set(common[train_count:])

    # move paired files
    for base in train_basenames:
        shutil.move(f"{txt_directory}/{base}.txt", f"{train_directory}/labels/{base}_{video_number}.txt")
        shutil.move(f"{jpg_directory}/{base}.jpg", f"{train_directory}/images/{base}_{video_number}.jpg")

    for base in val_basenames:
        shutil.move(f"{txt_directory}/{base}.txt", f"{val_directory}/labels/{base}_{video_number}.txt")
        shutil.move(f"{jpg_directory}/{base}.jpg", f"{val_directory}/images/{base}_{video_number}.jpg")

