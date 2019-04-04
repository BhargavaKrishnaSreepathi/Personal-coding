import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from matplotlib import animation


data = pd.read_excel(r'C:\Users\krish\Documents\GitHub\Personal-coding\data-visualization\GDP_data_cleaned.xlsx')

fig,ax = plt.subplots()
# ax = plt.axes()
# line, = ax.plot([], [], lw=2)

# def init():
#     line.set_data([], [])
#     return line,

def data_function(i):
    country_names = data['Country Name'].values
    GDP = data[str(i)].values

    df = pd.DataFrame({
        'country_names': country_names,
        'GDP': GDP
    })

    df = df.sort_values(by='GDP', ascending=False)
    df = df[:15]
    # line.set_data(df['country_names'], df['GDP'])
    return df

df = data_function(2015)
# anim = animation.FuncAnimation(fig, data_function, init_func=init,
#                                frames=100, interval=20, blit=True)
df['GDP_bn'] = round(df['GDP']/1000000000.0)

plt.barh(df['country_names'], df['GDP_bn'])
# plt.yticks(y_pos, objects)
# plt.xlabel('Usage')
ax.xaxis.set_visible(False)

for i, v in enumerate(df['GDP_bn']):
    ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold',horizontalalignment='left',
                verticalalignment='top')
plt.title('GDP figures')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), frameon=False)

plt.show()