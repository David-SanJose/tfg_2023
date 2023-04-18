import os
import pandas as pd
import shutil

folder = "..\\imagenes\\img2\\Aumentadas"
dest_folder = "..\\imagenes\\img2\\Aumentadas_test"
results = [each for each in os.listdir(folder) if each.endswith('.png')]

#convert list to DataFrame
df = pd.DataFrame(results, columns=["nombre"])

df_test = df.sample(frac=0.2, random_state=0)

for img_name in list(df_test.nombre):
    labels_name = img_name[:-3]+"txt"

    prev_img_path = folder + "\\" + img_name
    prev_labels_path = folder + "\\" + labels_name

    dest_img_path = dest_folder + "\\" + img_name
    dest_labels_path = dest_folder + "\\" + labels_name
    print(prev_img_path, prev_labels_path)
    print(dest_img_path, dest_labels_path)
    print("-"*10)

    shutil.move(prev_img_path, dest_img_path)
    shutil.move(prev_labels_path, dest_labels_path)


