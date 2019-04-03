import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from matplotlib import animation


data = pd.read_excel(r'C:\Users\krish\Documents\GitHub\Personal-coding\data-visualization\GDP_data_cleaned.xlsx')

fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def data_function(i):
    country_names = data['Country Name'].values
    GDP = data[str(i)].values

    df = pd.DataFrame({
        'country_names': country_names,
        'GDP': GDP
    })

    df = df.sort_values(by='GDP', ascending=False)
    df = df[:15]
    line.set_data(df['country_names'], df['GDP'])
    return df

df = data_function(2015)
anim = animation.FuncAnimation(fig, data_function, init_func=init,
                               frames=100, interval=20, blit=True)

# plt.barh(df['country_names'], df['GDP'])
# # plt.yticks(y_pos, objects)
# # plt.xlabel('Usage')
# plt.title('Programming language usage')

plt.show()