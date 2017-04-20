


import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
get_ipython().magic(u'matplotlib inline')
import time
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode
import requests
import pandas as pd
from scipy.spatial.distance import euclidean
import datetime
get_ipython().magic(u'matplotlib inline')
from dateutil.parser import parse
plt.style.use('ggplot')

url = "https://www.google.com/flights/explore/"
driver = webdriver.Chrome()

driver = webdriver.Chrome(desired_capabilities=ua, service_args=['--ignore-ssl-errors=true'])

driver.implicitly_wait(20) #waits for 20 seconds before following the execution up with the next line of code, to allow DOM to find unavailable elements
driver.get(url)

driver.save_screenshot(r'flight_explorer.png')

s = BeautifulSoup(driver.page_source, "lxml")

#Task 1

def scrape_data(start_date, from_place, to_place, city_name):
    #Saving the URL containing all but the date
    DateURL = driver.current_url[:-10]

    prices = []
    dates = []

    #Converting argument start_date to a str, Task 1.1
    startdate_str = start_date.strftime("%Y-%m-%d")
    DateURL = DateURL + startdate_str
    driver.get(DateURL) #We now have updated the URL to contain the date as required
    time.sleep(2)

    #Task 1.2
    from_input = driver.find_elements_by_css_selector('.LJTSM3-p-a')[0].click()
    elem = ActionChains(driver)
    elem.send_keys(from_place)
    elem.send_keys(Keys.ENTER) #We've now entered the argument 'from place' into the textbox in the URL
    elem.perform()
    time.sleep(2)

    #Task 1.3
    to_inputwo = driver.find_elements_by_css_selector('.LJTSM3-p-a')[1].click()
    elem = ActionChains(driver)
    elem.send_keys(to_place)
    elem.send_keys(Keys.ENTER) #We've now entered the argument 'to_place' into the textbox in the URL
    elem.perform()
    time.sleep(2)

    #Retrieving information about the desired city
    cities = driver.find_elements_by_css_selector('.LJTSM3-v-d')
    x = [cit.text for cit in cities]
    y = [unidecode(t) for t in x]
    z = [c.lower() for c in y]
    for index in range(len(z)):
        if city_name.lower() in z[index]:
            test = cities[index]
    #Making a DataFrame for dates and prices
    bars = test.find_elements_by_class_name("LJTSM3-w-x")
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.01)
        price = test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text
        date = test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text
        prices.append(price)
        dates.append(date)
    df = pd.DataFrame({'Date of Flight': dates, 'Price': prices})
    return df

#Task 2

def scrape_data_90(start_date, from_place, to_place, city_name):
    #Saving the URL containing all but the date
    DateURL = driver.current_url[:-10]

    prices = []
    dates = []

    #Converting argument start_date to a str, Task 2.1
    startdate_str = start_date.strftime("%Y-%m-%d")
    DateURL = DateURL + startdate_str
    driver.get(DateURL) #We now have updated the URL to contain the date as required
    time.sleep(2)

    #Task 2.2
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()
    elem = ActionChains(driver)
    elem.send_keys(from_place)
    elem.send_keys(Keys.ENTER) #We've now entered the argument 'from place' into the textbox in the URL
    elem.perform()
    time.sleep(2)

    #Task 2.3
    to_inputwo = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_inputwo.click()
    elem = ActionChains(driver)
    elem.send_keys(to_place)
    elem.send_keys(Keys.ENTER) #We've now entered the argument 'to_place' into the textbox in the URL
    elem.perform()
    time.sleep(2)

    #Retrieving information about the desired city
    cities = driver.find_elements_by_class_name('LJTSM3-v-c')
    x = [cit.text for cit in cities]
    y = [unidecode(t.lower()) for t in x]
    z = [c.lower() for c in y]
    for index in range(len(z)):
        if city_name.lower() in z[index]:
            c_index = index
    cities2 = driver.find_elements_by_class_name('LJTSM3-v-d')
    test = cities2[c_index]
    time.sleep(2)
    #Making a DataFrame for dates and prices
    bars = test.find_elements_by_class_name("LJTSM3-w-x")
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.01)
        price = test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text
        date = test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text
        prices.append(price)
        dates.append(date)
    time.sleep(2)
    nxt = driver.find_element_by_class_name('LJTSM3-w-D')
    ActionChains(driver).move_to_element(nxt).perform()
    time.sleep(2)
    nxt.click()
    time.sleep(2)
    for barb in bars[30:60]:
        ActionChains(driver).move_to_element(barb).perform()
        time.sleep(0.01)
        data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
                    test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
        clean_data = [(parse(d[1].split('-')[0].strip()), float(d[0].replace('$', '').replace(',', '')))
                      for d in data]
    df = pd.DataFrame(clean_data, columns=['start_date', 'price'])
    return df

x = scrape_data(datetime.datetime(2017, 4, 20), 'Scandinavia', 'NYC', 'Copenhagen')

clean_data = [(parse(d[1].split('-')[0].strip()), float(d[0].replace('$', '').replace(',', '')))
                      for d in data]
df = pd.DataFrame(clean_data, columns('start_date', 'price'))

def task_3_dbscan(flight_data):
    df = pd.DataFrame(columns=["start_date", "price"])
    df["price"] = flight_data['price']
    days = np.arange(len(df["price"]))
    df["start_date"] = days
    X = StandardScaler().fit_transform(df[["start_date", "price"]])
    db = DBSCAN(eps=.7, min_samples=5).fit(X)
    #Seems to work best with the above epsilon
    df['dbscan_labels'] = db.labels_
    outliers = (df['dbscan_labels'] == -1)
    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12, 8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14,
              y=1.01)
    dist = []
    lbls = np.unique(labels)
    print "Cluster labels: {}".format(np.unique(lbls))
    cluster_means = [np.mean(X[labels == num, :], axis=0) for num in range(lbls[-1] + 1)]
    print "Cluster Means: {}".format(cluster_means)
    outlier_indices = []
    for i, f in enumerate(outliers):
        if f == True:
            outlier_indices.append(i)
    noise_point_coord = []
    for item in outlier_indices:
        noise_point_coord.append(X[item, :])
    dist = []
    distlist = []
    for t, n in enumerate(noise_point_coord):
        for cm in cluster_means:
            dist.append(euclidean(n, cm))
        distlist.insert(t, dist)
        dist = []
    print "Euclidean distance: {}".format(distlist)
    print ("outlier indices:", outlier_indices)
    u = []
    list3 = []
    list4 = []
    mean = []
    std = []
    for i in distlist:
        u=(i.index(min(i)))
        for key, h in enumerate(df['dbscan_labels']):

            if u == h:
                list3.append(df['price'][key])
        list4.append(list3)
        list3 = []
    print ("Nearest Cluster-Wise Prices:", list4)
    for z in list4:
        mean.append(np.mean(z))
        std.append(np.std(z))
    print ("Means and Standard Deviations:", mean, std)
    list5 = []
    for g in outlier_indices:
        list5.append(df['price'][g])
    print ("outlier prices", list5)
    outlierprices = pd.DataFrame(columns=('start_date','price'))
    for m, j in enumerate(list5):
        count = 0
        if (j < max(mean[m] - (2*std[m]),50)):
            outlierprices.iloc[count] = df.iloc[outlier_indices[m]]
            count += 1
    print outlierprices
    
#Task 3.2 - Finding the Inter-Quartile Range
def task_3_IQR(flight_data):
    df = flight_data.copy()
    a = np.array(df['Price'])
    q3 = np.percentile(a, 75) # return 75th percentile
    q1 = np.percentile(a,25)
    iqr = q3 - q1
    lrange = (q1-1.5)*iqr
    for pr in df['Price']:
        if pr < lrange:
            print("We have", x, "outlier(s) and good price(s).")
            x =+ 1
        else:
            print(pr, "is not an outlier price.")
    plt.boxplot(df['Price'])
    plt.title('Boxplot depicting IQR')
    plt.savefig('task_3_iqr.png')
