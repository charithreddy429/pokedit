from PIL import Image
import glob
print("start")
# Path to the folder containing PNG files
folder_path = r"E:\CHAR\python\temp\img\ct"

# Use glob to find all PNG files in the folder
png_files = glob.glob(folder_path + "/**/*.png")


def resize(path):
    # Load the image
    image = Image.open(path)
    image = image.convert("RGBA")
    # Get the size of the image
    width, height = image.size

    # Calculate the new size
    new_width = 700
    new_height = int(height * (700 / width))
    if not(new_height<=1330):
        new_height = 1330
        new_width = int(width * (1330 / height))
    print(new_height,new_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with a white background
    new_image = Image.new("RGBA", (1080, 1920), (0,0,0,0))

    # Calculate the position to paste the resized image
    x = int((1080 - new_width) / 2)
    y = int((1330 - new_height/2) )

    # Paste the resized image onto the new image
    # new_image.paste(resized_image, (x, y))
    new_image.paste(resized_image, (x, y), mask=resized_image.split()[3])
    # Save the new image
    new_image.save(path.replace("ct","re"))
    


if __name__ == '__main__':
    x=0
    for png_file in png_files:
        if x>4:
            break
        resize(png_file)
        x+=1
        print(x)