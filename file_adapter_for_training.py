import os
from sklearn.model_selection import train_test_split

ruta_rgb = "..\\imagenes\\img2\\RGB"
files_list = os.listdir(ruta_rgb)

rgb_list = []
tag_list = []

for file in files_list:
    if file.endswith(".png"):
        tag_file_name = f"{file[:-3]}txt"

        if os.path.exists(f"{ruta_rgb}\\{tag_file_name}"):
            rgb_list.append(file)
            tag_list.append(tag_file_name)
        else:
            print("NO existe:", tag_file_name)

train_images, val_images, train_annotations, val_annotations = train_test_split(
    rgb_list, tag_list, test_size = 0.2, random_state = 1
    )

print(len(train_annotations), len(train_images))
print(len(val_annotations), len(val_images))

print(train_annotations[:4])
print(train_images[:4])