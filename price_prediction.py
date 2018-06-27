import requests,math
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from dateutil.parser import parse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
import random
from bokeh.plotting import figure, output_file, show

pd.set_option('display.max_columns', 30)
pd.set_option('display.line_width', 200)

# Function to read data from csv file
def read_data_csv(file_name):
    data_df = pd.read_csv(file_name)
    print("************ data from csv ************")
    print(data_df.head())
    return data_df

# Function to make dynamic requests to get data
def read_data_api(ticker_name, start_date, end_date, freq_in_min):
    headers = {'Content-Type': 'application/json',
               'Authorization': 'Token 243605654ef881d04ecd512109c9244d54486a9d'}
    # get_url = "https://api.tiingo.com/iex/VMW/prices?startDate=2017-05-01&endDate=2017-05-1&resampleFreq=1min"
    print("start_date : ",start_date)
    print("end_date   : ",end_date)
    get_url = "https://api.tiingo.com/iex/"+ticker_name+"/prices?startDate="+start_date+"&endDate="+end_date+"&resampleFreq="+str(freq_in_min)
    requestResponse = requests.get(get_url, headers=headers)
    print("requestResponse : ",requestResponse)
    df = pd.DataFrame(requestResponse.json())
    print("response dataframe : ",df.head())

    # Saving data to csv file
    df.to_csv("full_year_ndaq_data.csv")

    records = json.loads(df.T.to_json()).values()
    # Saving Data to mongodb
    for event in records:
        print("event : ",event)
        event["ticker"] = ticker_name
        event["frequency"] = freq_in_min
        event["flag"] = "5year"
        # inserting
        db.insert(event)

    return df

# Function to get data from mongodb
def read_data_db(database, ticker_name, start_date, end_date, frequency):
    # splitted_sd = start_date.split("-")
    # splitted_ed = end_date.split("-")
    # s_date = datetime.datetime(int(splitted_sd[0]),int(splitted_sd[1]),int(splitted_sd[2]))
    # e_date = datetime.datetime(int(splitted_ed[0]),int(splitted_ed[1]),int(splitted_ed[2]))
    # datetime(start_date)
    print('getting data from mongodb for {} from {} to {} with frequecy of {}'.format(ticker_name, start_date, end_date, frequency))
    data = database.find({
                          'ticker':ticker_name,
                           # 'date':{"$gt":start_date, "$lte":end_date},
                           'frequency':frequency
                          })

    out = pd.DataFrame(list(data))
    print("total data points : ",len(out))
    return out

#
def draw_plot(dataframe):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 2, 1)
    ax.set_xlabel('Date')
    # ax.set_ylabel('Closing price ($)')
    # dataframe.date = dataframe.date.apply(lambda x: x.split("T")[0])
    ax.plot(dataframe.date, dataframe.close, label=ticker)
    ax.plot(dataframe.date, dataframe.short_avg, label='Short Avg', color='green')
    ax.plot(dataframe.date, dataframe.long_avg, label='Long Avg', color='red')
    # ax.legend()
    # plt.xticks(rotation=90)
    # plt.savefig(ticker+'_stocks.png')
    draw_gold(fig)
    plt.show()
    return fig

def date_time(x):
    return np.array(x, dtype=np.datetime64)

def draw_plot_bokeh(dataframe):
    p1 = figure(x_axis_type="datetime", title="{} Closing Prices".format(ticker), plot_width=1000, plot_height=500)
    p1.grid.grid_line_alpha = 0.3
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price'
    p1.line(date_time(dataframe.date), dataframe.close, color='blue', legend="close")
    p1.line(date_time(dataframe.date), dataframe.short_avg, color='green', legend="short_avg")
    p1.line(date_time(dataframe.date), dataframe.long_avg, color='red', legend="long_avg")
    p1.legend.location = "top_left"
    # output to static HTML file
    output_file("lines.html")
    show(p1)
    return p1


def draw_gold(fig):
    dataframe = pd.read_csv('GOLD_small.csv')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('Date')
    # ax.set_ylabel('Closing price ($)')
    # dataframe.date = dataframe.date.apply(lambda x: x.split("T")[0])
    ax1.plot(dataframe.date, dataframe.close, label=ticker)
    # ax.legend()
    # plt.xticks(rotation=90)
    # plt.savefig(ticker + '_stocks.png')
    # plt.show()
    return fig


# Function to draw plot using matplotlib
def find_intersection_points(first_list, second_list, date_list, fig):
    # f = np.array(dataframe.close.values)
    # g = np.array(dataframe.short_avg.values)
    f = np.array(first_list)
    g = np.array(second_list)
    print("************** f **************",f)
    print("************** g **************",g)
    idx = np.argwhere(np.diff(np.sign(f - g)) != 0).reshape(-1) + 0
    plt.plot(date_list[idx], g[idx], 'ro', color='blue')
    # plt.xticks(rotation=90)
    # plt.show()
    return idx, fig

#
def find_adjacent_close_prices(plot_list_1,plot_list_2, indices):
    print("*************** Finding Adjacent Indices ******************")
    rising_indices = []
    falling_indices = []

    for i,indx in enumerate(indices):
        curr_price = plot_list_1[indx]
        if i>0:
            prev_price = plot_list_1[indx - 1]
        else:
            prev_price = 0
        futr_price = plot_list_1[indx + 1]
        futr_avg = plot_list_2[indx + 1]
        # print(dataframe.date[indx], ", ", indx, ", ", curr_price, ", ", futr_price, ", ", prev_price)
        if futr_price >= curr_price and futr_price>=futr_avg:
        # if curr_price>futr_price:
            rising_indices.append(indx)
        elif futr_price<curr_price and futr_price>=futr_avg:
            rising_indices.append(indx)
        else:
            falling_indices.append(indx)
            # print("falling")

    print("falling_indices : ", falling_indices)
    print("rising_indices  : ", rising_indices)

    # if ticker=="NDAQ":
    return falling_indices, rising_indices
    # else:
    #     return falling_indices, rising_indices

# Finding Adjaccent Close Prices
def calculate_rolling_avgs(dataframe):
    print(dataframe.head())
    print("********* Calculating 50 Days rolling average *********")
    print("date : ")
    print(dataframe.keys())
    dataframe.date = dataframe.date.apply(lambda x: parse(x).strftime('%Y-%m-%d'))
    print("date now : ")
    print(dataframe.head())
    dataframe.close = dataframe.close.apply(lambda x: float(x) if x!='null' else 0)
    short_rolling_avg = dataframe.close.rolling(50).mean()
    long_rolling_avg = dataframe.close.rolling(200).mean()
    dataframe['short_avg'] = short_rolling_avg
    dataframe['long_avg'] = long_rolling_avg

    # dataframe.short_avg = dataframe.short_avg.apply(lambda x: float(x) if x!='null' else 0)
    print("************************ dataframe ********************")
    dataframe.short_avg = dataframe.short_avg.fillna(0)
    dataframe.long_avg = dataframe.long_avg.fillna(0)
    print(dataframe.tail())

    # print("********************** Calling Demo Plot ******************")
    # fig = draw_plot(dataframe)
    fig = draw_plot_bokeh(dataframe)

    indices, fig = find_intersection_points(first_list=dataframe.short_avg.values, second_list=dataframe.long_avg.values, date_list=dataframe.date, fig=fig)
    print("***************** Intersection Point *******************")
    print(indices)

    #find first index
    first_index = check_first_index(dataframe.short_avg.values,indices)
    # Function to find fall and rise indices
    fall_indices, rise_indices = generate_rise_and_fall(indices,first_index)

    # Finding Rising and Falling Indices
    fall_indices, rise_indices = find_adjacent_close_prices(dataframe.short_avg.values, dataframe.long_avg.values, indices)

    # Ploting Rising and Falling Indices
    plot_rise_and_fall_bokeh(dataframe.date.values, dataframe.short_avg.values, rise_indices, fall_indices, fig)

    # # Finding Short Windows using rising indices and falling indices
    # windows = get_windows(dataframe.date.values, dataframe.short_avg.values, rise_indices, fall_indices)
    #
    # # Function to find matrices from short windows
    # matrices = get_matrices(dataframe, windows)
    # plt.show()
    # return dataframe

# Function to check first index
def check_first_index(plot_list, indices):
    indx = indices[0]
    futr_price = plot_list[indx + 1]
    curr_price = plot_list[indx]
    if futr_price>curr_price:
        return "rise"
    else:
        return 'fall'

# Function to generate rise indices and falling indices
def generate_rise_and_fall(arry, first_index=""):
    if first_index == "rise":
        rise_indices = [arry[i] for i in range(0,len(arry),2)]
        fall_indices = [arry[i] for i in range(1,len(arry),2)]
    else:
        fall_indices = [arry[i] for i in range(0, len(arry), 2)]
        rise_indices = [arry[i] for i in range(1, len(arry), 2)]
    print("rising indices  : ",rise_indices)
    print("falling indices  : ",fall_indices)
    return fall_indices,rise_indices

# Plotting Rising and Falling Indices
def plot_rise_and_fall(date_list, plot_list, rise_indx, fall_indx, fig):
    plt.plot(date_list[rise_indx], plot_list[rise_indx], 'ro', color='green')
    plt.plot(date_list[fall_indx], plot_list[fall_indx], 'ro', color='red')
    # plt.show()

# Plotting Rising and Falling Indices
def plot_rise_and_fall_bokeh(date_list, plot_list, rise_indx, fall_indx, fig):
    fig.circle(date_list[rise_indx], plot_list[rise_indx],color='green')
    fig.circle(date_list[fall_indx], plot_list[fall_indx],color='red')
    show(fig)

# Function to find the windows using rising points
def get_windows(date_list, plot_list, rising_indices, falling_indices):
    windows = []
    windows_out = []
    print("rising_indices : ",len(rising_indices))
    print("falling_indices: ",len(falling_indices))
    for i in range(len(rising_indices)):
        try:
            if falling_indices[i+1]:
                windows.append((rising_indices[i], falling_indices[i]))
        except:
            windows.append((rising_indices[i], len(date_list)-1))
            pass
    print("************** Short Windows ***************")
    print(windows)
    colors = ["red", "blue", "black", "cyan", "magenta", "green", "grey"]

    for i in range(len(windows)):
        d_indx = [j for j in range(windows[i][0], windows[i][1])]
        windows_out.append(d_indx)
        plt.plot(date_list[d_indx], plot_list[d_indx], color=random.choice(colors))

    # plt.show()
    return windows_out

# Function to get matrices from the window frames
def get_matrices(dataframe, windows):
    print("**************** Getting Insight from the windows *******************")
    print()
    # dataframe.date = dataframe.date.apply(lambda x : x.split("T")[0])
    rise_days_array = []
    fall_days_array = []
    rise_days_percents = []
    fall_days_percents = []
    out_array = []
    for window in windows:
        out = {}
        close_prices = dataframe.close[window].values
        dates = dataframe.date[window].values
        max_close = np.max(close_prices)
        min_close = np.min(close_prices)
        mean_close = np.mean(close_prices)
        mean_close = round(mean_close,3)
        std_close = np.std(close_prices, ddof=1)
        std_close = round(std_close,3)
        n_days = len(window)
        maxp_day = dataframe.date[window[list(close_prices).index(max_close)]]
        minp_day = dataframe.date[window[list(close_prices).index(min_close)]]
        max_rise_diff = window[list(close_prices).index(max_close)] - window[list(close_prices).index(min_close)]
        max_fall_diff = n_days - max_rise_diff
        rise_percentage = round((max_close-min_close)/max_close * 100,3)
        fall_percentage = round((max_close-close_prices[-1])/max_close * 100,3)
        rise_days_percents.append(rise_percentage)
        fall_days_percents.append(fall_percentage)
        rise_days_array.append((n_days, max_rise_diff))
        fall_days_array.append((n_days, max_fall_diff))
        out["entry_to_peak"] = rise_percentage
        out["peak_to_exit"] = fall_percentage
        out["days_peak_to_exit"] = max_fall_diff
        out["days_to_peak"] = max_rise_diff
        out["exit_50"] = close_prices[-1]
        out["mean_price"] = mean_close
        out["entry_50"] = min_close
        out["peak_price"] = max_close
        out["total_days"] = n_days
        out["std_dev"] = std_close

        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(1, 1, 1)
        title = ticker+" Stocks between "+dates[0]+" to "+dates[-1]
        filename = title+".png"
        ax.set_title(str(title))
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing price ($)')
        ax.plot(dates, close_prices, label=ticker)
        ax.plot(maxp_day, max_close, 'ro', label='peak_price : '+str(max_close), color='green')
        ax.plot(minp_day, min_close, 'ro', label='entry_50 : '+str(min_close), color='red')
        ax.plot(dates, [mean_close]*len(dates), label='mean_price : '+str(mean_close), color='cyan')
        ax.plot(dates, [std_close+max_close]*len(dates), 'ro', label='std_dev   : '+str(std_close), color='pink')
        ax.plot(dates, [std_close+max_close]*len(dates), 'ro', label='total_days  : '+str(n_days), color='magenta')
        ax.plot(dates, [std_close+max_close]*len(dates), 'ro', label='days_to_peak : '+str(max_rise_diff), color='blue')
        ax.plot(dates, [std_close+max_close]*len(dates), 'ro', label='days_peak_to_exit : '+str(max_fall_diff), color='blue')
        ax.plot(dates, [std_close+max_close]*len(dates), 'ro', label='entry_to_peak % : '+str(rise_percentage), color='blue')
        ax.plot(dates, [std_close+max_close]*len(dates), 'ro', label='peak_to_exit % : '+str(fall_percentage), color='blue')

        ax.legend(loc="upper left")
        plt.xticks(rotation=90)
        plt.savefig(filename)
        out_array.append(out)
        plt.show()

    print("average window frame : ",np.mean([x[0] for x in rise_days_array]))
    print("average rise   days  : ",np.mean([x[1] for x in rise_days_array]))
    print("average fall   days  : ",np.mean([x[1] for x in fall_days_array]))
    print("average rise percent : ",np.mean(rise_days_percents))
    print("average fall percent : ",np.mean(fall_days_percents))
    print()
    print("********************* Metrics **************************")
    out = pd.DataFrame(out_array)
    # out.to_csv("distribution.csv")
    print(out)

    # Previous Data
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Days')
    # ax.set_ylabel('Frequency')
    # ax.set_title("Days to Peak Frequency Distribution")
    # data = list(out.days_to_peak.values)
    # n, bins, patches = plt.hist(data, 10, histtype='bar', rwidth=0.8, stacked=True, label='days_to_peak')

    fig, ax = plt.subplots(figsize=(14, 5))
    data = list(out.days_to_peak.values)
    make_hist(ax, data, bins=list(range(10)) + list(range(10, 41, 5)) + [np.inf], extra_y=6,
              title='Days to Peak Frequency Distribution', xlabel='Days', ylabel='Frequency')
    # plt.show()

    x = list(out.entry_to_peak.values)
    fig, ax = plt.subplots(figsize=(14, 5))
    make_hist(ax, x, bins=list(range(10)) + list(range(10, 41, 5)) + [np.inf], extra_y=6,title='Rise Percentages Frequency Distribution',xlabel='Percentages',ylabel='Frequency')
    plt.show()

# Draw CandleSticks
def draw_candlesticks(dataframe):
    dataframe.date = dataframe.date.apply(lambda x : mdates.datestr2num(x))
    # dataframe.date = dataframe.date.apply(bytespdate2num('%Y-%m-%d'))
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    open = dataframe.open.values
    close = dataframe.close.values
    high = dataframe.high.values
    low = dataframe.low.values
    date = dataframe.date.values

    ohlc = []
    x=0
    y = len(date)
    while x < y:
        append_me = date[x], float(open[x]), float(high[x]), float(low[x]), float(close[x])
        ohlc.append(append_me)
        x += 1
    candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.grid(True)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(ticker)
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    # plt.show()
    return fig

# Make Histogram Function
def make_hist(ax, x, bins=None, binlabels=None, width=0.85, extra_x=1, extra_y=4,
              text_offset=0.3, title=r"Frequency diagram",
              xlabel="Values", ylabel="Frequency"):
    if bins is None:
        xmax = max(x)+extra_x
        bins = range(xmax+1)
    if binlabels is None:
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else
                         '{}-{}'.format(bins[i], bins[i+1]-1)
                         for i in range(len(bins)-1)]
        else:
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else
                         '{}-{}'.format(*bins[i:i+2])
                         for i in range(len(bins)-1)]
        if bins[-1] == np.inf:
            binlabels[-1] = '{}+'.format(bins[-2])
    n, bins = np.histogram(x, bins=bins)
    patches = ax.bar(range(len(n)), n, align='center', width=width)
    ymax = max(n)+extra_y

    ax.set_xticks(range(len(binlabels)))
    ax.set_xticklabels(binlabels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis='y')
    # http://stackoverflow.com/a/28720127/190597 (peeol)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # http://stackoverflow.com/a/11417222/190597 (gcalmettes)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    autolabel(patches, text_offset)
    fname=ticker+"_"+title+".png"
    plt.savefig(fname)

def autolabel(rects, shift=0.3):
    """
    http://matplotlib.org/1.2.1/examples/pylab_examples/barchart_demo.html
    """
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            plt.text(rect.get_x()+rect.get_width()/2., height+shift, '%d'%int(height),
                     ha='center', va='bottom')

if __name__ == '__main__':
    # db = MongoClient('localhost', 27017)["algo_data"]["tiingo_test"]

    # ticker = input("please enter ticker name (ex. NDAQ): ")
    # st_date = input("please enter start date (ex. 2017-05-01): ")
    # ed_date = input("please enter end date (ex. 2017-05-01): ")
    # frequency = input("please enter granular frequency (ex. 1min): ")

    ticker = 'AAPL.csv'
    st_date = "2013-04-01"
    ed_date = '2014-04-01'

    # st_date=datetime(2017, 1, 1, 0, 0, 0)
    # ed_date=datetime(2018, 4, 1, 0, 0, 0)

    frequency = "12hour"

    # Function to read data using tiingo.com api
    # data_df = read_data_api(ticker_name=ticker, start_date=st_date, end_date=ed_date, freq_in_min=frequency)

    # Function to read data from a csv or excel file
    # data_df = read_data_csv('TSLA.csv')
    # data_df = read_data_csv('ADSK.csv')
    data_df = read_data_csv(ticker)

    # Function to read data from mongodb
    # data_df = read_data_db(database=db, ticker_name=ticker, start_date=st_date, end_date=ed_date, frequency=frequency)

    # Function to plot the graph using plotly
    # draw_graph(data_df=data_df)

    # Function to calculate 50days rolling average
    data_df_with_avgs = calculate_rolling_avgs(data_df)