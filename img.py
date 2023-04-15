import urllib.request
import tqdm
import os

script_path = os.getcwd()

# URL of the image to be downloaded
url = "https://assets.pokemon.com/assets/cms2/img/pokedex/detail/"

for i in tqdm.tqdm(range(1,52)):
    # Full path of the file where the image should be saved
    filename = script_path+r"\ik"+f"\\{i}.png"
    nurl = url + "{:03d}".format(i)+".png"
    # Download the image from the URL and save it to the specified file
    urllib.request.urlretrieve(nurl, filename)
