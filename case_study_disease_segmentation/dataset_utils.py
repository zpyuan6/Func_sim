import os
import shutil


# In this case study, we build an image segmentation dataset based on multispectral UAV images from Peach tree disease dataset (https://ieee-dataport.org/documents/peach-tree-disease-detection-dataset). 
def split_task_dataset(source_dataset, target_path):
    image_path = os.path.join(source_dataset, "input")
    label_path = os.path.join(source_dataset, "label")

    image_ids = []
    task_ids = []

    for root, folders, files in os.walk(image_path):
        for file in files:
            image_id = file.split(".")[0]
            image_ids.append(image_id)

            task_id = "_".join(image_id.split("_")[0:3])

            if not task_id in task_ids:
                os.makedirs(os.path.join(target_path, task_id, "input"))
                os.makedirs(os.path.join(target_path, task_id, "label"))
                task_ids.append(task_id)

            shutil.copyfile(os.path.join(root,file), os.path.join(target_path, task_id, "input",  file))
            shutil.copyfile(os.path.join(label_path,file), os.path.join(target_path, task_id, "label",  file))

if __name__ == "__main__":
    
    original_dataset_path = "F:\Hyperspecial\pear_processed\segmentation_data"
    life_long_dataset_path = "F:\Hyperspecial\pear_processed\life_long_dataset"

    split_task_dataset(original_dataset_path, life_long_dataset_path)


