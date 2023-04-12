from PIL import Image
import glob
print("start")
# Path to the folder containing PNG files
folder_path = r"E:\CHAR\python\temp\img\ik"

# Use glob to find all PNG files in the folder
png_files = glob.glob(folder_path + "/**/*.png")


def cut(path):

    # Load the image
    image = Image.open(path).convert("RGBA")
    # Get the image size
    width, height = image.size
    # Initialize the leftmost, rightmost, topmost, and bottommost pixel coordinates to the image width and height respectively
    leftmost = width
    rightmost = 0
    topmost = height
    bottommost = 0
    # Loop over all the pixels in the image
    for x in range(width):
        for y in range(height):
            # Get the RGBA values of the pixel
            r, g, b, a = image.getpixel((x, y))
            
            # If the pixel is not transparent
            if a != 0:
                # Update the leftmost, rightmost, topmost, and bottommost pixel coordinates if necessary
                if x < leftmost:
                    leftmost = x
                if x > rightmost:
                    rightmost = x
                if y < topmost:
                    topmost = y
                if y > bottommost:
                    bottommost = y
                    
    # Crop the image to the leftmost, rightmost, topmost, and bottommost pixels
    cropped_image = image.crop((leftmost, topmost, rightmost+1, bottommost+1))
    # Save the cropped image to a file

    cropped_image.save(path.replace("ik","re"))
x=0
for png_file in png_files:
    if x>4:
        break
    cut(png_file)
    x+=1
    print(x)