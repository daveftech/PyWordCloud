import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv('test_data/winemag-data-130k-v2.csv')
print(df.head())

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))

print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()), ", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()), ", ".join(df.country.unique()[0:5])))

print(df[['country', 'description', 'points']].head())

by_country = df.groupby('country')

print(by_country.describe().head())

print(by_country.mean().sort_values(by = "points", ascending = False).head())

# Generate some plots from the data, to explore it a bit

# plt.figure(figsize=(15, 10))
# by_country.size().sort_values(ascending = False).plot.bar()
# plt.xticks(rotation = 50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Number of Wines")
# plt.show()

# plt.figure(figsize=(15,10))
# by_country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Highest point of Wines")
# plt.show()

# Generate a basic word cloud
# text = df.description[0]
# wordcloud = WordCloud().generate(text)

# Try our real text
f = open("data/2109_speech_trudeau.txt")
text = f.read()
f.close()

wordcloud = WordCloud(background_color = 'white').generate(text)

plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

f = open("data/2019_speech_sheer.txt")
text = f.read()
f.close()

wordcloud = WordCloud(background_color = 'white').generate(text)

plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()