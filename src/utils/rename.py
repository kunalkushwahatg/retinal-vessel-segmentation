import os
import shutil

# Define the source and destination directories
source_dir = '/home/kunalkushwahatg/Desktop/rentinal_research/data/DRHAGIS/output'
destination_dir = '/home/kunalkushwahatg/Desktop/rentinal_research/data/DRHAGIS/output_renamed'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('_manual_orig.png'):
        # Construct the new filename
        new_filename = filename.split('_manual_orig.png')[0] + '.png'
        
        # Construct the full file paths
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, new_filename)
        
        # Copy and rename the file
        shutil.copy(source_file, destination_file)

print("Files have been renamed and copied successfully.")
