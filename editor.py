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
def pop_lerp(x: float) -> float:
    if x <= .50:
        return (2.1 * x) ** 2 + 0.1
    elif x <= 1:
        return ((2.1 * (1-x)) ** 2) * (2 - 2 * x) + 1 * (2 * x - 1) + 0.1
    else:
        return 1


def lpop_lerp(x: float) -> float:
    if x <= .50:
        return min((2.1 * x) ** 2 + 0.1, 1)
    elif x <= 1:
        return min(((2.1 * (1-x)) ** 2) * (2 - 2 * x) + 1 * (2 * x - 1) + 0.1, 1)
    else:
        return 1


def popeffect(img: np.ndarray, x: float, eff=pop_lerp) -> np.ndarray:
    new_shape = (int(img.shape[1] * eff(x)), int(img.shape[0] * eff(x)))
    img = cv2.resize(img, new_shape)  # type: ignore
    return img


def blit(bg: np.ndarray, img: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    alpha = img[:, :, 3] / 255.0
    y1, x1 = pos
    y2, x2 = pos + np.array(img.shape[0:2])
    # Paste img onto bg using the alpha channel as a mask
    bg[y1:y2, x1:x2, 0] = bg[y1:y2, x1:x2, 0] * (1 - alpha) + img[:, :, 0] * alpha
    bg[y1:y2, x1:x2, 1] = bg[y1:y2, x1:x2, 1] * (1 - alpha) + img[:, :, 1] * alpha
    bg[y1:y2, x1:x2, 2] = bg[y1:y2, x1:x2, 2] * (1 - alpha) + img[:, :, 2] * alpha
    return bg


def change_size(img: np.ndarray) -> np.ndarray:
    width, height = img.shape[0:2]
    new_width = 700
    new_height = int(height * (700 / width))
    if not(new_height <= 750):
        new_height = 750
        new_width = int(width * (750 / height))
    return cv2.resize(img, (new_height, new_width))


def cut(image:np.ndarray)->np.ndarray:
    # Get the image size
    width, height = image.shape[0:2]
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
            r, g, b, a = image[x, y]

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
    cropped_image = image[leftmost: rightmost + 1, topmost: bottommost + 1]

    return cropped_image


def neg(img:np.ndarray)->np.ndarray:
    # Extract the alpha channel
    alpha = img[:, :, 3]

    # Create a mask for non-transparent pixels
    mask = alpha > 0

    # Replace non-transparent pixels with black
    img[mask] = [0, 0, 0, 255]

    # Convert the NumPy array back to an image
    return img


def paste(image:np.ndarray, neg_img:np.ndarray, out_path: str) ->None:
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


def create_vid(image:np.ndarray,neg:np.ndarray,out_path:str)->None:
    def calculate_frame1(frame,x):
        poped = popeffect(neg,6*x)
        frame = blit(frame,poped,(1296,540)-(0.5*np.array(poped.shape[0:2])).astype(int))# type: ignore
        return frame
    def calculate_frame2(x):
        frame = np.full((1920, 1080, 4), (255, 255, 255,255), dtype=np.uint8)
        popedi = popeffect(image,6*x)
        poped = popeffect(np.full((192, 1080, 4),(0,0,0,255),dtype=np.uint8),12*x,lpop_lerp)
        frame = blit(frame,poped,(96,540)-(0.5*np.array(poped.shape[0:2])).astype(int))# type: ignore
        frame = blit(frame,poped,(1824,540)-(0.5*np.array(poped.shape[0:2])).astype(int)) # type: ignore
        frame = blit(frame,popedi,(1296,540)-(0.5*np.array(popedi.shape[0:2])).astype(int))# type: ignore
        return frame[:,:,0:3]
    neg = change_size(neg)
    image = change_size(image)
    cap = cv2.VideoCapture(BG_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    counter = 0
    while cap.isOpened():
        counter +=1
        ret, frame = cap.read()

        if not ret:
            break
        frame = calculate_frame1(frame,counter/frame_count)
        out.write(frame)
    counter =0
    for i in range(120):
        counter+=1
        out.write(calculate_frame2(counter/120))
    cap.release()
    out.release()


def addsound(vid_path: str, sound_paths: list, start_times: list, output_file: str,bg_music=None)->None:
    with VideoFileClip(vid_path) as video:
        final_clip = None  # type: ignore
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
            final_clip:VideoClip = concatenate_videoclips([final_clip, clip])
        if bg_music:
            bg_clip = AudioFileClip(bg_music).subclip(0,final_clip.duration)
            final_audio = CompositeAudioClip([final_clip.audio, bg_clip])
            final_clip.audio = final_audio
        final_clip.write_videofile(output_file)


def create_whole(num:int)->None:
    img = cv2.imread(f"ik\\{num}.png",cv2.IMREAD_UNCHANGED)
    ct = cut(img)
    ne = neg(ct.copy())
    create_vid(ct,ne,f"vid\\vid{num}.mp4")


if __name__ == "__main__":
    for num in tqdm.tqdm(range(25, 30)):
        create_whole(num)
        audioarr = ["assets\\sounds\\guess.mp3",
                    save_speech(f"it's {names[num - 1]}", f"assets\\sounds\\{num}.mp3"),
                    "assets\\sounds\\plz.mp3"]
        addsound(f"vid\\vid{num}.mp4", audioarr, [0, 3.1, 5.2], f"vid\\vid{num}t.mp4")
