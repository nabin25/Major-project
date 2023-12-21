import os

# Specify the folder path where your files are located
folder_path = "E:/major-project/pre-images/pre-for-all2"

# Set the starting value for renaming
starting_value = 3664  # Change this to your desired starting value

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter for files that start with "hand_" and have a four-digit number
file_list = [file for file in file_list if file.startswith('hand_') and file[5:9].isdigit()]

# Sort the files by their numeric part
file_list.sort(key=lambda x: int(x[5:9]))

# Rename the files with sequential numbers starting from the specified value
for i, file_name in enumerate(file_list):
    new_name = f'hand_{starting_value + i:04d}.png'  # Replace 'extension' with the actual file extension
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)
    
    # Avoid collisions by checking if the new name already exists
    if not os.path.exists(new_path):
        os.rename(old_path, new_path)

print("Files have been renamed.")
