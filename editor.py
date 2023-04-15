from PIL import Image
import numpy as np
import cv2
# import os
import tqdm
from moviepy.editor import *
from voice import save_speech
from names import *

script_path = os.getcwd()

BG_PATH = r'assets\background\background12.mp4'


def cut(image):
    # Get the image size
    width, height = image.size
    # Initialize the leftmost, rightmost, topmost, and bottommost pixel
    # coordinates to the image width and height respectively
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
    if not (new_height <= 1330):
        new_height = 1330
        new_width = int(width * (1330 / height))
    print(new_height, new_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with a white background
    new_image = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))

    # Calculate the position to paste the resized image
    x = int((1080 - new_width) / 2)
    y = int((1330 - new_height / 2))

    # Paste the resized image onto the new image
    # new_image.paste(resized_image, (x, y))
    new_image.paste(resized_image, (x, y), mask=resized_image.split()[3])
    # Save the new image

    return new_image


def paste(image, neg_img, out_path: str) -> ():
    # Open the video file
    cap = cv2.VideoCapture(BG_PATH)
    bg_img = np.array(neg_img)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # create constants

    extra_duration = 5

    background = Image.new('RGB', image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    background = np.array(background)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    height = background.shape[0]

    # Calculate the top and bottom cutoff points
    top_cutoff = int(0.1 * height)
    bottom_cutoff = int(0.9 * height)

    # Set the top and bottom 10% of the image to black
    background[0:top_cutoff, :] = [0, 0, 0]
    background[bottom_cutoff:, :] = [0, 0, 0]
    # Loop over the frames of the video
    alpha = bg_img[:, :, 3] / 255.0
    alpha_complement = 1 - alpha

    bg_img0 = bg_img[:, :, 0] * alpha
    bg_img1 = bg_img[:, :, 1] * alpha
    bg_img2 = bg_img[:, :, 2] * alpha

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Paste bg_img onto frame using the alpha channel as a mask
        frame[:, :, 0] = frame[:, :, 0] * alpha_complement + bg_img0
        frame[:, :, 1] = frame[:, :, 1] * alpha_complement + bg_img1
        frame[:, :, 2] = frame[:, :, 2] * alpha_complement + bg_img2

        # Write the frame to output video file
        out.write(frame)
    for _ in range(int(fps * extra_duration)):
        out.write(background)

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def addsound(vid_path: str, sound_paths: list, start_times: list, output_file: str):
    with VideoFileClip(vid_path) as video:
        final_clip = None
        last_end_time = 0
        for i in range(len(sound_paths)):
            sound_path = sound_paths[i]
            start_time = start_times[i]
            end_time = video.duration

            if i < len(sound_paths) - 1:
                end_time = start_times[i + 1]

            if start_time > last_end_time:
                # create a new clip with the section of the video
                # between the end of the last clip and the start of
                # the current clip
                clip = video.subclip(last_end_time, start_time)
                if final_clip is None:
                    final_clip = clip
                else:
                    final_clip = concatenate_videoclips([final_clip, clip])

            # add the audio clip
            audio_clip = AudioFileClip(sound_path)
            if audio_clip.duration < end_time - start_time:
                end_time = start_time + audio_clip.duration
            audio_clip = audio_clip.subclip(0, end_time - start_time)
            video_clip = video.subclip(start_time, end_time)
            video_clip = video_clip.set_audio(audio_clip)
            if final_clip is None:
                final_clip = video_clip
            else:
                final_clip = concatenate_videoclips([final_clip, video_clip])
            last_end_time = end_time

        # add the final section of the video
        if last_end_time < video.duration:
            clip = video.subclip(last_end_time, video.duration)
            final_clip = concatenate_videoclips([final_clip, clip])

        final_clip.write_videofile(output_file)

    def create_whole(num):
        print(num)
        img = Image.open(f"ik\\{num}.png")
        ct = cut(img)
        re = resize(ct)
        ne = neg(re)
        ct.save(f"re\\{num}c.png")
        re.save(f"re\\{num}.png")
        ne.save(f"re\\{num}n.png")
        paste(f"re\\{num}.png", f"re\\{num}n.png", f"vid\\vid{num}.mp4")


def create_whole(num):
    img = Image.open(f"ik\{num}.png")
    ct = cut(img)
    re = resize(ct)
    ne = neg(re)
    paste(re, ne, f"vid\vid{num}.mp4")


if __name__ == "__main__":

    for num in tqdm.tqdm(range(1, 3)):
        create_whole(num)
        audioarr = ["assets\\sounds\\guess.mp3",
                    save_speech(f"it's {names[num - 1]}", f"assets\\sounds\\{num}.mp3"),
                    "assets\\sounds\\plz.mp3"]
        addsound(f"vid\\vid{num}.mp4", audioarr, [0, 3.1, 5.2], f"vid\\vid{num}t.mp4")
