import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, lognorm, chisquare
from tabulate import tabulate #pretty print of tables. source: http://txt.arboreus.com/2013/03/13/pretty-print-tables-in-python.html
import warnings
warnings.filterwarnings('ignore')

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"


# Functions for exploratory data analysis
def visualize_continuous_variables(df, label, method={'type': 'histogram', 'bins': 20}, outlier='on'):
    """
    function to quickly visualize continous variables
    df: pandas.dataFrame
    label: str, name of the variable to be plotted. It should be present in df.columns
    method: dict, contains info of the type of plot to generate. It can be histogram or boxplot [-Not yet developped]
    outlier: {'on','off'}, Set it to off if you need to cut off outliers. Outliers are all those points
    located at 3 standard deviations further from the mean
    """
    # create vector of the variable of interest
    v = df[label]
    # define mean and standard deviation
    m = v.mean()
    s = v.std()
    # prep the figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].set_title('Distribution of ' + label)
    ax[1].set_title('Tip % by ' + label)
    if outlier == 'off':  # remove outliers accordingly and update titles
        v = v[(v - m) <= 3 * s]
        ax[0].set_title('Distribution of ' + label + '(no outliers)')
        ax[1].set_title('Tip % by ' + label + '(no outliers)')
    if method['type'] == 'histogram':  # plot the histogram
        v.hist(bins=method['bins'], ax=ax[0])
    if method['type'] == 'boxplot':  # plot the box plot
        df.loc[v.index].boxplot(label, ax=ax[0])
    ax[1].plot(v, df.loc[v.index].tip_percentage, '.', alpha=0.4)
    ax[0].set_xlabel(label)
    ax[1].set_xlabel(label)
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Tip (%)')

def visualize_categories(df, catName, chart_type='histogram', ylimit=[None, None]):
    """
    This functions helps to quickly visualize categorical variables.
    This functions calls other functions generate_boxplot and generate_histogram
    df: pandas.Dataframe
    catName: str, variable name, it must be present in df
    chart_type: {histogram,boxplot}, choose which type of chart to plot
    ylim: tuple, list. Valid if chart_type is histogram
    """
    print (catName)
    cats = sorted(pd.unique(df[catName]))
    if chart_type == 'boxplot':  # generate boxplot
        generate_boxplot(df, catName, ylimit)
    elif chart_type == 'histogram':  # generate histogram
        generate_histogram(df, catName)
    else:
        pass

    # => calculate test statistics
    groups = df[[catName, 'tip_percentage']].groupby(catName).groups  # create groups
    tips = df.tip_percentage
    if len(cats) <= 2:  # if there are only two groups use t-test
        print (ttest_ind(tips[groups[cats[0]]], tips[groups[cats[1]]]))
    else:  # otherwise, use one_way anova test
        # prepare the command to be evaluated
        cmd = "f_oneway("
        for cat in cats:
            cmd += "tips[groups[" + str(cat) + "]],"
        cmd = cmd[:-1] + ")"
        print ("one way anova test:", eval(cmd))  # evaluate the command and print
    print ("Frequency of categories (%):\n", df[catName].value_counts(normalize=True) * 100)

def test_classification(df, label, yl=[0, 50]):
    """
    This function test if the means of the two groups with_tip and without_tip are different at 95% of confidence level.
    It will also generate a box plot of the variable by tipping groups
    label: str, label to test
    yl: tuple or list (default = [0,50]), y limits on the ylabel of the boxplot
    df: pandas.DataFrame (default = data)

    Example: run <visualize_continuous(data,'Fare_amount',outlier='on')>
    """

    if len(pd.unique(df[label])) == 2:  # check if the variable is categorical with only two  categores and run chisquare test
        vals = pd.unique(df[label])
        gp1 = df[df.with_tip == 0][label].value_counts().sort_index()
        gp2 = df[df.with_tip == 1][label].value_counts().sort_index()
        print ("t-test if", label, "can be used to distinguish transaction with tip and without tip")
        print (chisquare(gp1, gp2))
    elif len(pd.unique(df[label])) >= 10:  # other wise  run the t-test
        df.boxplot(label, by='with_tip')
        plt.ylim(yl)
        plt.show()
        print ("t-test if", label, "can be used to distinguish transaction with tip and without tip")
        print ("results:", ttest_ind(df[df.with_tip == 0][label].values, df[df.with_tip == 1][label].values, False))
    else:
        pass


def generate_boxplot(df, catName, ylimit):
    """
    generate boxplot of tip percentage by variable "catName" with ylim set to ylimit
    df: pandas.Dataframe
    catName: str
    ylimit: tuple, list
    """
    df.boxplot('tip_percentage', by=catName)
    # plt.title('Tip % by '+catName)
    plt.title('')
    plt.ylabel('Tip (%)')
    if ylimit != [None, None]:
        plt.ylim(ylimit)
    plt.show()


def generate_histogram(df, catName):
    """
    generate histogram of tip percentage by variable "catName" with ylim set to ylimit
    df: pandas.Dataframe
    catName: str
    ylimit: tuple, list
    """
    cats = sorted(pd.unique(df[catName]))
    colors = plt.cm.jet(np.linspace(0, 1, len(cats)))
    hx = np.array(map(lambda x: round(x, 1), np.histogram(df.tip_percentage, bins=20)[1]))
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    for i, cat in enumerate(cats):
        vals = df[df[catName] == cat].tip_percentage
        h = np.histogram(vals, bins=hx)
        w = 0.9 * (hx[1] - hx[0]) / float(len(cats))
        plt.bar(hx[:-1] + w * i, h[0], color=colors[i], width=w)
    plt.legend(cats)
    plt.yscale('log')
    plt.title('Distribution of Tip by ' + catName)
    plt.xlabel('Tip (%)')


def exploring_data(data):

    # Print the size of the dataset
    print("Number of rows:", data.shape[0])
    print("Number of columns: ", data.shape[1])
    #
    # # define the figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    # histogram of the number of trip distance
    data.trip_distance.hist(bins=30, ax=ax[0])
    ax[0].set_xlabel('Trip Distance (miles)')
    ax[0].set_ylabel('Count')
    ax[0].set_yscale('log')
    ax[0].set_title('Histogram of Trip Distance with outliers included')

    # create a vector to contain Trip Distance
    v = data.trip_distance
    # exclude any data point located further than 3 standard deviations of the median point and
    # plot the histogram with 30 bins
    v[~((v - v.median()).abs() > 3 * v.std())].hist(bins=30, ax=ax[1])  #
    ax[1].set_xlabel('Trip Distance (miles)')
    ax[1].set_ylabel('Count')
    ax[1].set_title('A. Histogram of Trip Distance (without outliers)')

    # apply a lognormal fit. Use the mean of trip distance as the scale parameter
    scatter, loc, mean = lognorm.fit(data.trip_distance.values,
                                     scale=data.trip_distance.mean(),
                                     loc=0)
    pdf_fitted = lognorm.pdf(np.arange(0, 12, .1), scatter, loc, mean)
    ax[1].plot(np.arange(0, 12, .1), 6000000 * pdf_fitted, 'r')
    ax[1].legend(['data', 'lognormal fit'])

    # export the figure
    plt.savefig('Question2.jpeg', format='jpeg')
    # plt.show()

    #
    # First, convert pickup and drop off datetime variable in their specific righ format
    data.rename(columns={'lpep_pickup_datetime': 'Pickup_dt', 'lpep_dropoff_datetime': 'Dropoff_dt'}, inplace=True)
    data['Pickup_dt'] = data.Pickup_dt.apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"))
    data['Dropoff_dt'] = data.Dropoff_dt.apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"))

    # Second, create a variable for pickup hours
    data['Pickup_hour'] = data.Pickup_dt.apply(lambda x: x.hour)

    # Mean and Median of trip distance by pickup hour
    # I will generate the table but also generate a plot for a better visualization

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))  # prepare fig to plot mean and median values
    # use a pivot table to aggregate Trip_distance by hour
    table1 = data.pivot_table(index='Pickup_hour', values='trip_distance', aggfunc=('mean', 'median')).reset_index()
    # rename columns
    table1.columns = ['Hour', 'Mean_distance', 'Median_distance']
    table1[['Mean_distance', 'Median_distance']].plot(ax=ax)
    plt.ylabel('Metric (miles)')
    plt.xlabel('Hours after midnight')
    plt.title('Distribution of trip distance by pickup hour')
    # plt.xticks(np.arange(0,30,6)+0.35,range(0,30,6))
    plt.xlim([0, 23])
    plt.savefig('Question3_1.jpeg', format='jpeg')
    # plt.show()
    print('-----Trip distance by hour of the day-----\n')
    print(tabulate(table1.values.tolist(), ["Hour", "Mean distance", "Median distance"]))
    #
    # # select airport trips
    airports_trips = data[(data.RatecodeID == 2) | (data.RatecodeID == 3)]
    print("Number of trips to/from NYC airports: ", airports_trips.shape[0])
    print("Average fare (calculated by the meter) of trips to/from NYC airports: $", airports_trips.fare_amount.mean(),
          "per trip")
    print("Average total charged amount (before tip) of trips to/from NYC airports: $",
          airports_trips.total_amount.mean(), "per trip")
    #
    # create a vector to contain Trip Distance for
    v2 = airports_trips.trip_distance  # airport trips
    v3 = data.loc[~data.index.isin(v2.index), 'trip_distance']  # non-airport trips

    # remove outliers:
    # exclude any data point located further than 3 standard deviations of the median point and
    # plot the histogram with 30 bins
    v2 = v2[~((v2 - v2.median()).abs() > 3 * v2.std())]
    v3 = v3[~((v3 - v3.median()).abs() > 3 * v3.std())]

    # define bins boundaries
    bins = np.histogram(v2, normed=True)[1]
    h2 = np.histogram(v2, bins=bins, normed=True)
    h3 = np.histogram(v3, bins=bins, normed=True)

    # plot distributions of trip distance normalized among groups
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    w = .4 * (bins[1] - bins[0])
    ax[0].bar(bins[:-1], h2[0], alpha=1, width=w, color='b')
    ax[0].bar(bins[:-1] + w, h3[0], alpha=1, width=w, color='g')
    ax[0].legend(['Airport trips', 'Non-airport trips'], loc='best', title='group')
    ax[0].set_xlabel('trip distance (miles)')
    ax[0].set_ylabel('Group normalized trips count')
    ax[0].set_title('A. trip distance distribution')
    # ax[0].set_yscale('log')

    # plot hourly distribution
    airports_trips.Pickup_hour.value_counts(normalize=True).sort_index().plot(ax=ax[1])
    data.loc[~data.index.isin(v2.index), 'Pickup_hour'].value_counts(normalize=True).sort_index().plot(ax=ax[1])
    ax[1].set_xlabel('Hours after midnight')
    ax[1].set_ylabel('Group normalized trips count')
    ax[1].set_title('B. Hourly distribution of trips')
    ax[1].legend(['Airport trips', 'Non-airport trips'], loc='best', title='group')
    plt.savefig('Question3_2.jpeg', format='jpeg')
    # plt.show()

    return data

def clean_data(adata):
    """
    This function cleans the input dataframe adata:
    . drop Ehail_fee [99% transactions are NaNs]
    . impute missing values in Trip_type
    . replace invalid data by most frequent value for RateCodeID and Extra
    . encode categorical to numeric
    . rename pickup and dropff time variables (for later use)

    input:
        adata: pandas.dataframe
    output:
        pandas.dataframe

    """
    ## make a copy of the input
    data = adata.copy()
    ## drop Ehail_fee: 99% of its values are NaNs
    if 'ehail_fee' in data.columns:
        data.drop('ehail_fee', axis=1, inplace=True)

    ##  replace missing values in Trip_type with the most frequent value 1
    data['trip_type'] = data['trip_type'].replace(np.NaN, 1)

    ## replace all values that are not allowed as per the variable dictionary with the most frequent allowable value
    # remove negative values from Total amound and Fare_amount
    print ("Negative values found and replaced by their abs")
    print ("total_amount", 100 * data[data.total_amount < 0].shape[0] / float(data.shape[0]), "%")
    print ("fare_amount", 100 * data[data.fare_amount < 0].shape[0] / float(data.shape[0]), "%")
    print ("improvement_surcharge", 100 * data[data.improvement_surcharge < 0].shape[0] / float(data.shape[0]), "%")
    print ("tip_amount", 100 * data[data.tip_amount < 0].shape[0] / float(data.shape[0]), "%")
    print ("tolls_amount", 100 * data[data.tolls_amount < 0].shape[0] / float(data.shape[0]), "%")
    print ("mta_tax", 100 * data[data.mta_tax < 0].shape[0] / float(data.shape[0]), "%")
    data.total_amount = data.total_amount.abs()
    data.fare_amount = data.fare_amount.abs()
    data.improvement_surcharge = data.improvement_surcharge.abs()
    data.tip_amount = data.tip_amount.abs()
    data.tolls_amount = data.tolls_amount.abs()
    data.mta_tax = data.mta_tax.abs()

    # RateCodeID
    indices_oi = data[~((data.RatecodeID >= 1) & (data.RatecodeID <= 6))].index
    data.loc[indices_oi, 'RatecodeID'] = 1  # 2 = Cash payment was identified as the common method
    print (round(100 * len(indices_oi) / float(data.shape[0]), 2), "% of values in RateCodeID were invalid.--> Replaced by the most frequent 2")

    # Extra
    indices_oi = data[~((data.extra == 0) | (data.extra == 0.5) | (data.extra == 1))].index
    data.loc[indices_oi, 'extra'] = 0  # 0 was identified as the most frequent value
    print (round(100 * len(indices_oi) / float(data.shape[0]), 2), "% of values in Extra were invalid.--> Replaced by the most frequent 0")

    # Total_amount: the minimum charge is 2.5, so I will replace every thing less than 2.5 by the median 11.76 (pre-obtained in analysis)
    indices_oi = data[(data.total_amount < 2.5)].index
    data.loc[indices_oi, 'total_amount'] = 11.15
    print (round(100 * len(indices_oi) / float(data.shape[0]), 2), "% of values in total amount worth <$2.5.--> Replaced by the median 11.15")

    # encode categorical to numeric (I avoid to use dummy to keep dataset small)
    if data.store_and_fwd_flag.dtype.name != 'int64':
        data['store_and_fwd_flag'] = (data.store_and_fwd_flag == 'Y') * 1

    print ("Done cleaning")
    return data

def engineer_features(adata):
    """
    This function create new variables based on present variables in the dataset adata. It creates:
    . Week: int {1,2,3,4,5}, Week a transaction was done
    . Week_day: int [0-6], day of the week a transaction was done
    . Month_day: int [0-30], day of the month a transaction was done
    . Hour: int [0-23], hour the day a transaction was done
    . Shift type: int {1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)}, shift of the day
    . Speed_mph: float, speed of the trip
    . Tip_percentage: float, target variable
    . With_tip: int {0,1}, 1 = transaction with tip, 0 transction without tip

    input:
        adata: pandas.dataframe
    output:
        pandas.dataframe
    """

    # make copy of the original dataset
    data = adata.copy()

    # derive time variables
    print ("deriving time variables...")
    ref_week = datetime.datetime(2017, 1, 1).isocalendar()[1]  # first week of september in 2015
    data['Week'] = data.Pickup_dt.apply(lambda x: x.date().isocalendar()[1]) - ref_week + 1
    data['Week_day'] = data.Pickup_dt.apply(lambda x: x.date().isocalendar()[2])
    data['Month_day'] = data.Pickup_dt.apply(lambda x: x.day)
    data['Hour'] = data.Pickup_dt.apply(lambda x: x.hour)
    data['Month'] = data.Pickup_dt.apply(lambda x: x.month)


    # data.rename(columns={'Pickup_hour':'Hour'},inplace=True)

    # create shift variable:  1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)
    data['Shift_type'] = np.NAN
    data.loc[data[(data.Hour >= 7) & (data.Hour < 15)].index, 'Shift_type'] = 1
    data.loc[data[(data.Hour >= 15) & (data.Hour < 23)].index, 'Shift_type'] = 2
    data.loc[data[data.Shift_type.isnull()].index, 'Shift_type'] = 3

    # Trip duration
    print ("deriving Trip_duration...")
    data['trip_duration'] = ((data.Dropoff_dt - data.Pickup_dt).apply(lambda x: x.total_seconds() / 60.))

    # create variable for Speed
    print ("deriving Speed. Make sure to check for possible NaNs and Inf vals...")
    data['speed_mph'] = data.trip_distance / (data.trip_duration / 60)
    # replace all NaNs values and values >240mph by a values sampled from a random distribution of
    # mean 12.9 and  standard deviation 6.8mph. These values were extracted from the distribution
    indices_oi = data[(data.speed_mph.isnull()) | (data.speed_mph > 240)].index
    data.loc[indices_oi, 'speed_mph'] = np.abs(np.random.normal(loc=12.9, scale=6.8, size=len(indices_oi)))
    print ("Feature engineering done! :-)")

    # create tip percentage variable
    data['tip_percentage'] = 100 * data.tip_amount / data.total_amount

    # create with_tip variable
    data['with_tip'] = (data.tip_percentage > 0) * 1

    return data

if __name__ == '__main__':
    # Download the 2017 taxi trip data dataset
    data = pd.read_csv(r'C:\Users\krish/Downloads/2017_Green_Taxi_Trip_Data.csv')

    data = exploring_data(data)

    data = clean_data(data)

    # run the code to create new features on the dataset
    print("size before feature engineering:", data.shape)
    data = engineer_features(data)
    print("size after feature engineering:", data.shape)

    data_train = data[(data.Pickup_dt <= '2017-01-31 23:59:59')]
    print (data_train.shape)
    data_month = data[data.Month == 1]
    print (data_month.shape)
    ## code to compare the two Tip_percentage identified groups
    # split data in the two groups
    data1 = data[data.tip_percentage > 0]
    data2 = data[data.tip_percentage == 0]

    # generate histograms to compare
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    data.tip_percentage.hist(bins=20, normed=True, ax=ax[0])
    ax[0].set_xlabel('Tip (%)')
    ax[0].set_title('Distribution of Tip (%) - All transactions')

    data1.tip_percentage.hist(bins=20, normed=True, ax=ax[1])
    ax[1].set_xlabel('Tip (%)')
    ax[1].set_title('Distribution of Tip (%) - Transaction with tips')
    ax[1].set_ylabel('Group normed count')
    plt.savefig('Question4_target_varc.jpeg', format='jpeg')
    # plt.show()


    # Example of exploration of the Fare_amount using the implented code:
    visualize_continuous_variables(data1, 'fare_amount', outlier='on')
    # test_classification(data,'fare_amount',[0,25])

    continuous_variables = ['total_amount', 'fare_amount', 'trip_distance', 'trip_duration', 'tolls_amount',
                            'speed_mph', 'tip_percentage']
    cor_mat = data1[continuous_variables].corr()
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    plt.imshow(cor_mat)
    plt.xticks(range(len(continuous_variables)), continuous_variables, rotation='vertical')
    plt.yticks(range(len(continuous_variables)), continuous_variables)
    plt.colorbar()
    plt.title('Correlation between continuous variables')
    plt.show()
