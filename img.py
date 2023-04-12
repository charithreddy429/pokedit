import urllib.request
import tqdm

# URL of the image to be downloaded
url = "https://assets.pokemon.com/assets/cms2/img/pokedex/detail/"

for i in tqdm.tqdm(range(1,30)):
    # Full path of the file where the image should be saved
    filename = r"E:\CHAR\python\temp\img\ik\\"+f"{int(i/100)}\\"+f"{i%100}.png"
    nurl = url + "{:03d}".format(i)+".png"
    # Download the image from the URL and save it to the specified file
    urllib.request.urlretrieve(nurl, filename)
