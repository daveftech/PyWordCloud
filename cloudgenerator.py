import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# Set up the stopwords
stopwords = set(STOPWORDS)
stopwords.update(["us", "will"])

# Read in the data files
f = open("data/2019_speech_trudeau.txt", encoding = "utf8")
trudeau_text = f.read()
f.close()

f = open("data/2019_speech_sheer.txt", encoding = "utf8")
sheer_text = f.read()
f.close()

f = open("data/2019_speech_singh.txt", encoding = "utf8")
singh_text = f.read()
f.close()

# Load the mask
mask = np.array(Image.open("masks/canada_mapleleaf_mask.png"))
image_colors = ImageColorGenerator(mask)

wc_trudeau = WordCloud(stopwords = stopwords, background_color = "white", mode = "RGBA", max_words = 1000, mask = mask).generate(trudeau_text)
wc_sheer = WordCloud(stopwords = stopwords, background_color = "white", mode = "RGBA", max_words = 1000, mask = mask).generate(sheer_text)
wc_singh = WordCloud(stopwords = stopwords, background_color = "white", mode = "RGBA", max_words = 1000, mask = mask).generate(sheer_text)

# Set up the subplots and display them
fig, axes = plt.subplots(2, 3)


axes[0][0].imshow(wc_trudeau.recolor(color_func=image_colors), interpolation="bilinear")
axes[0][0].axis('off')
axes[0][0].set_title("Election Speech: Trudeau")

axes[0][1].imshow(mask, interpolation = "bilinear")
axes[0][1].axis('off')

axes[0][2].imshow(wc_sheer.recolor(color_func=image_colors), interpolation="bilinear")
axes[0][2].axis('off')
axes[0][2].set_title("Election Speech: Sheer")

axes[1][1].imshow(wc_singh.recolor(color_func=image_colors), interpolation="bilinear")
axes[1][1].axis('off')
axes[1][1].set_title("Election Speech: Singh")

axes[1][0].axis('off')
axes[1][2].axis('off')

plt.show()