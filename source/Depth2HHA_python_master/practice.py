from PIL import Image
import numpy as np

image = np.array(Image.open("./demo/0_raw.png"))

print(f"{image.shape=}")
print(f"{image.max()=}")
print(f"{image.min()=}")
