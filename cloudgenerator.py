import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# Read in the data files
f = open("data/2019_speech_trudeau.txt")
trudeau_text = f.read()
f.close()

f = open("data/2019_speech_sheer.txt")
sheer_text = f.read()
f.close()

# Load the mask
mask = np.array(Image.open("masks/canada_mapleleaf_mask.png"))
image_colors = ImageColorGenerator(mask)

wc_trudeau = WordCloud(background_color = "white", mode = "RGBA", max_words = 1000, mask = mask).generate(trudeau_text)
wc_sheer = WordCloud(background_color = "white", mode = "RGBA", max_words = 1000, mask = mask).generate(sheer_text)

# Set up the subplots and display them
fig, axes = plt.subplots(1, 2)

axes[0].imshow(wc_trudeau.recolor(color_func=image_colors), interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Election Speech: Trudeau")

axes[1].imshow(wc_sheer.recolor(color_func=image_colors), interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Election Speech: Sheer")

plt.show()