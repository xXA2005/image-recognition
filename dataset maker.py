import os
from base64 import b64encode
import requests

apikey = "your api key here"

try:
    os.mkdir("./dataset/other/")
except:
    pass

for folder in os.listdir("./images"):
    for img in os.listdir("./images/" + folder):
        imgbytes = open("./images/" + folder + "/" + img, "rb").read()
        imgb64 = b64encode(imgbytes).decode("utf-8")
        r = requests.post("https://free.nocaptchaai.com/solve", json={
            "images":  {"0": imgb64},
            "target":  folder,
            "method":  "hcaptcha_base64",
            "ln":      "en",
            "type":    "grid",
            "choices": [],
        }, headers={"content-type": "application/json",
                    "apikey": apikey
                    })
        js = r.json()
        while True:
            if js["status"] == "solved":
                try:
                    open(f"./dataset/{js['target']}/{img}",
                         "wb").write(imgbytes)
                except FileNotFoundError:
                    os.mkdir(f"./dataset/{js['target']}")
                    open(f"./dataset/{js['target']}/{img}",
                         "wb").write(imgbytes)
                break
            elif js["status"] == "skip":
                open(f"./dataset/other/{img}", "wb").write(imgbytes)
                break
            elif js["status"] == "new":
                js = requests.get(js["url"]).json()
                print(js)
                continue
            else:
                print(js)
                break
        print("Image: " + img)
