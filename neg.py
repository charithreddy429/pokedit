import numpy as np
from PIL import Image
import glob
print("start")
# Path to the folder containing PNG files
folder_path = r"E:\CHAR\python\temp\img\re"

# Use glob to find all PNG files in the folder
png_files = glob.glob(folder_path + "/**/*.png")

def neg(path):
    # Load the image and convert to RGBA format
    img = Image.open(path).convert('RGBA')

    # Convert the image to a NumPy array
    arr = np.array(img)

    # Extract the alpha channel
    alpha = arr[:, :, 3]

    # Create a mask for non-transparent pixels
    mask = alpha > 0

    # Replace non-transparent pixels with black
    arr[mask] = [0, 0, 0, 255]

    # Convert the NumPy array back to an image
    result = Image.fromarray(arr, mode='RGBA')
    result.save(path.replace("re","neg"))

if __name__ == '__main__':
    x=0
    for png_file in png_files:
        if x>4:
            break
        neg(png_file)
        x+=1
        print(x)