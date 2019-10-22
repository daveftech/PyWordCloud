import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# Read in the data file
df = pd.read_csv('test_data/winemag-data-130k-v2.csv')

# Group the dataframe by country
by_country = df.groupby('country')

# Combine the text for all the reviews for canada
canada = " ".join(review for review in df[df["country"] == "Canada"].description)

# Load the mask
mask = np.array(Image.open("masks/canada_mapleleaf_mask.png"))

wordcloud = WordCloud(background_color = "white", mode = "RGBA", max_words = 1000, mask = mask).generate(canada)

image_colors = ImageColorGenerator(mask)
print(image_colors)

plt.figure(figsize=[7,7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()