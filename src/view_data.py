from PIL import Image

dataFile = open("optdigits.tra")
for line in dataFile:
    content = list(map(lambda i: int(i), line.split(',')[:64]))
    img = Image.new('RGB', (8,8), "white")
    data = [(int(i * 255 / 16), int(i * 255 / 16), int(i * 255 / 16)) for i in content]
    img.putdata(data)
    img.show()
    input()
