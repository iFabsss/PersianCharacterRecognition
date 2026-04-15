import os
import shutil
from tokenize import String

# CHANGE THIS to your parent folder path
persian_folder1 = "PersianAlphabetDataset"
persian_folder2 = "PersianAlphabetDataset2"
dataset_folder = "dataset"
images_folder = "dataset/images"
labels_folder = "dataset/labels"

# Ensure dataset folders exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# File to store all unique characters
character_list_file = os.path.join(dataset_folder, "Persian Characters List.txt")

# Store characters
characters = []

# Loop through each character folder
for folder_name in os.listdir(persian_folder1):
    folder_path = os.path.join(persian_folder1, folder_name)

    if os.path.isdir(folder_path):
        # Extract character name (after "-")
        try:
            _, character = folder_name.split("-", 1)
        except ValueError:
            print(f"Skipping invalid folder name: {folder_name}")
            continue

        # Capitalize character for text content
        character_text = character.capitalize()

        # Add to character list
        characters.append(character_text)

        # Loop through image files
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".jpg"):
                src_image_path = os.path.join(folder_path, file_name)
                dst_image_path = os.path.join(images_folder, file_name)

                # Copy image to image folder in dataset folder
                shutil.copy2(src_image_path, dst_image_path)

                # Create corresponding .txt filename
                txt_name = os.path.splitext(file_name)[0] + ".txt"
                txt_path = os.path.join(labels_folder, txt_name)

                # Write character name into text file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(character_text)

                print(f"Copied: {dst_image_path}")
                print(f"Created: {txt_path}")

i = 1

for folder_name in os.listdir(persian_folder2):
    folder_path = os.path.join(persian_folder2, folder_name)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".png"):
                try:
                    character, _ = file_name.split("_", 1)
                except ValueError:
                    print(f"Skipping invalid folder name: {file_name}")
                    continue

                character_text = character.capitalize()
                characters.append(character_text)

                new_file_name = f"{character_text}_{i}"
                src_image_path = os.path.join(folder_path, file_name)
                dst_image_path = os.path.join(images_folder, new_file_name + ".png")

                shutil.copy2(src_image_path, dst_image_path)

                txt_name = os.path.join(new_file_name + ".txt")
                txt_path = os.path.join(labels_folder, txt_name)

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(character_text)

                print(f"Copied: {dst_image_path}")
                print(f"Created: {txt_path}")

        i += 1


# Remove duplicates and sort
unique_characters = sorted(set(characters))

# Write all characters to a file
with open(character_list_file, "w", encoding="utf-8") as f:
    for char in unique_characters:
        f.write(char + "\n")

print("\nCharacter list saved to:", character_list_file)