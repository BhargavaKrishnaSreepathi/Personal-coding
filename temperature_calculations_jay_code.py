# importing packages to be used
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
from os import listdir
from os.path import isfile, join

# path to the folder in which all the csv files are present. Note that the folder should only have csv files and that
# which are related to the calculations
mypath = r'C:\Users\krish\Desktop\Jay COdes/filter_room_mar'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# loop for all the files present in the folder
for k in range(len(onlyfiles)):

    # 'onlyfiles' has the csv names of each files, so adding the path to call the csv file and saving as pandas dataframe
    df = pd.read_csv(r'C:\Users\krish\Desktop\Jay COdes/filter_room_mar/' + onlyfiles[k])

    # identifying the unique days present in the column 'D' and also the unique treatment groups present in column
    # 'Treatment'
    number_of_days = df.D.unique()
    number_of_treatment_groups = df.Treatment.unique()

    # initiating a dataframe to append the values of temperature into
    min_temp = []

    # looping for all the unique dates present in the csv file. Do note that the number_of_days stores a list of all
    # the unique dates present in the csv file
    for i in range(len(number_of_days)):

        # the current date in the loop
        current_date = number_of_days[i]
        print (current_date)

        # df1 is a subset of the df with filter applied on the 'D' column based on the current date
        df1 = df[df.D == current_date]

        # z is the temperature values of the current date present in the csv file
        z = df1.Temp.values

        # n_samples is the total length of z
        n_samples = len(z)

        # parameters regarding the ruptures algorithm. These are like rules of thumb, but will not exactly be followed
        # in the implementation of the algorithm, but rather used as pointers or guides
        dim = 1
        sigma = 4
        n_bkps = 6

        # implementation of the Pelt algorithm in the ruptures package whille applying breakpoints to z
        algo = rpt.Pelt(model="rbf").fit(z)

        # predicting the breakpoints of the z column. Penalty of 10, can be changed based on the amount of breakpoints
        # required. So if the penalty is made 1, the number of breakpoints will go to the range of 60 to 100. If the
        # penalty code is 100 the number of breakpoints will go to 1 to 5 range. penalty 10 was giving a range between
        # 5 to 15 breakpoints, depending on the z profile. There might be few exceptions based on the available data
        result = algo.predict(pen=10)
        print (result)

        # the script below helps in plotting the breakpoints on the z profile.
        indexes = range(len(z))
        plt.plot(indexes, z, 'ro')
        for xc in result:
            plt.axvline(x=xc)
        # uncomment the below line to save the plots. Do keep in mind the number of plots generated is equal to the
        # number of unique days present in the csv file. And in the bigger loop, this will be repeated for the total
        # number of rooms present in the folder
        # plt.savefig(r'C:\Users\krish\Desktop\Jay COdes/pdf/' + str(current_date) + '.pdf')
        plt.close()

        # initializing a list of the results of the breakpoints, with '0' being the first breakpoint
        a = [0] + result
        # initializing the minimum temperature as 50, to set a upper bound for minimum temperature. Do note that this
        # assumes that none of the temperatures in the csv files cross 50 Degrees
        min_temperature = 50

        # loop to identify the minimum temperature based on the breakpoints generated.
        # what this does is, take breakpoint i and breakpoint i+1.
        # Identify the temperatures of these corresponding breakpoints
        # add up all the temperatures in between these breakpoints
        # and then average them out by subtracting breakpoing i+1 and breakpoint i
        # this now represents the minimum average temperature in the breakpoints
        # this loop is repeated for all breakpoints and the minimum average temperature is saved for further processing
        for i in range(len(result)):
            temperature = (sum(z[a[i]:a[i+1]])) / (a[i+1] - a[i])
            if temperature < min_temperature:
                min_temperature = temperature

        min_temp.append([current_date, min_temperature])
    print (min_temp)

    # save all the minimum temperatures for all the days in a dataframe and then save it as a csv file with the same
    # name as the initial file for future use
    df_min_temp = pd.DataFrame(min_temp)
    df_min_temp.to_csv(r'C:\Users\krish\Desktop\Jay COdes/pdf/min_temp_' + onlyfiles[k])

