from PIL import Image
import numpy as np
import cv2
import os
import tqdm
script_path = os.getcwd()

BGPATH = r'assets\background12.mp4'
def cut(image):

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
    cropped_image = image.crop((leftmost, topmost, rightmost + 1, bottommost + 1))
    # Save the cropped image to a file

    return cropped_image
def neg(img):

    img = img.convert('RGBA')

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
    return result
def resize(image):
    # Load the image
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
    new_image.save(outpath)
    return new_image
def paste(imgpath,negpath,outpath):
    # Open the video file
    cap = cv2.VideoCapture(BGPATH)
    bg_img = cv2.imread(negpath, cv2.IMREAD_UNCHANGED)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    # Loop over the frames of the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        alpha = bg_img[:, :, 3] / 255.0

        # Paste bg_img onto frame using the alpha channel as a mask
        frame[:, :, 0] = frame[:, :, 0] * (1 - alpha) + bg_img[:, :, 0] * alpha
        frame[:, :, 1] = frame[:, :, 1] * (1 - alpha) + bg_img[:, :, 1] * alpha
        frame[:, :, 2] = frame[:, :, 2] * (1 - alpha) + bg_img[:, :, 2] * alpha

        # Write the frame to output video file
        out.write(frame)



    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":

    for i in tqdm.tqdm(range(1,10)):
        path = script_path+r"\ik"+f"\\{i}.png"
        outpath = script_path+r"\re"+f"\\{i}.png"
        ct = cut(Image.open(path))
        res = resize(ct)
        ne = neg(res)
        res.save(outpath)
        neg.save(outpath)