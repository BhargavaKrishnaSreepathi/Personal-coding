import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd

df = pd.read_csv(r'C:\Users\krish\Desktop\Jay COdes/filter_room_mar/dg_601_1_5_R.csv')

number_of_days = df.D.unique()
number_of_treatment_groups = df.Treatment.unique()

print (len(number_of_days))

print (len(number_of_treatment_groups))
print (df.Treatment.unique())
min_temp = []

for i in range(len(number_of_days)):
# for i in range(1):

    current_date = number_of_days[i]
    print (current_date)

    df1 = df[df.D == current_date]


    z = df1.Temp.values
    n_samples = len(z)
    dim = 1
    sigma = 4
    n_bkps = 6

    algo = rpt.Pelt(model="rbf").fit(z)
    result = algo.predict(pen=10)
    print (result)

    indexes = range(len(z))
    plt.plot(indexes, z, 'ro')
    for xc in result:
        plt.axvline(x=xc)
    plt.savefig(r'C:\Users\krish\Desktop\Jay COdes/pdf/' + str(current_date) + '.pdf')
    plt.close()
    a = [0] + result
    min_temperature = 50
    for i in range(len(result)):
        temperature = (sum(z[a[i]:a[i+1]])) / (a[i+1] - a[i])
        if temperature < min_temperature:
            min_temperature = temperature

    min_temp.append([current_date, min_temperature])
print (min_temp)


