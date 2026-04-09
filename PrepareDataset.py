import os

# CHANGE THIS to your parent folder path
parent_folder = "PersianAlphabetDataset"

# File to store all unique characters
character_list_file = os.path.join(parent_folder, "Persian Characters List.txt")

# Store characters
characters = []

# Loop through each character folder
for folder_name in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder_name)

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
                image_path = os.path.join(folder_path, file_name)

                # Create corresponding .txt filename
                txt_name = os.path.splitext(file_name)[0] + ".txt"
                txt_path = os.path.join(folder_path, txt_name)

                # Write character name into text file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(character_text)

                print(f"Created: {txt_path}")

# Remove duplicates and sort
unique_characters = sorted(set(characters))

# Write all characters to a file
with open(character_list_file, "w", encoding="utf-8") as f:
    for char in unique_characters:
        f.write(char + "\n")

print("\nCharacter list saved to:", character_list_file)