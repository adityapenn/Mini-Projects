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
import datetime
get_ipython().magic(u'matplotlib inline')
from dateutil.parser import parse
plt.style.use('ggplot')



url = "https://www.google.com/flights/explore/#explore;f=JFK,EWR,LGA;t=r-Scandinavia-0x4664bcea4d8e22cd%253A0x7186faed0155b381;li=3;lx=5;d=2017-04-10"
driver = webdriver.PhantomJS()

ua = dict(DesiredCapabilities.PHANTOMJS)
#Dict containing information about the headless browser


ua["phantomjs.page.settings.userAgent"] = ("Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36")
#Assign a newly created key the value of the user agent (my computer's user agent)

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
    to_input = driver.find_element_by_class_name("LJTSM3-s-a")
    to_input.click()
    elem = ActionChains(driver)
    elem.send_keys(from_place) 
    elem.send_keys(Keys.ENTER) #We've now entered the argument 'from place' into the textbox in the URL
    elem.perform() 
    time.sleep(2)
    
    #Task 1.3
    to_inputwo = driver.find_element_by_class_name("LJTSM3-s-b")
    to_inputwo.click()
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
    to_input = driver.find_element_by_class_name("LJTSM3-s-a")
    to_input.click()
    elem = ActionChains(driver)
    elem.send_keys(from_place) 
    elem.send_keys(Keys.ENTER) #We've now entered the argument 'from place' into the textbox in the URL
    elem.perform() 
    time.sleep(2)
    
    #Task 2.3
    to_inputwo = driver.find_element_by_class_name("LJTSM3-s-b")
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
    nxt.click()
    for barb in bars[30:60]:
        ActionChains(driver).move_to_element(barb).perform()
        time.sleep(0.01)
        priceb = test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text
        dateb = test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text
        prices.append(priceb)
        dates.append(dateb)
    df = pd.DataFrame({'Date of Flight': dates, 'Price': prices})
    return df


#scrape_data(datetime.datetime(2017, 4, 20), 'NYC', 'Scandinavia', 'Copenhagen')

#Task 3.1 
def task_3_dbscan(flight_data):

    X = np.concatenate([days[:, None], prices[:, None]], axis=1)
    db = DBSCAN(eps=.30, min_samples=4).fit(X)
    #Seems to work best with the above epsilon
    
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

    days = np.arange(60)
    prices1 = np.random.normal(0, 35, size=20) + 400
    prices2 = np.random.normal(0, 35, size=20) + 800
    prices3 = np.random.normal(0, 35, size=20) + 400
    prices = np.concatenate([prices1, prices2, prices3], axis=0)

    lbls = np.unique(db.labels_)
    cluster_means = [np.mean(X[labels == num, :], axis=0) for num in range(lbls[-1] + 1)]
    noise_point = X[30, :]

    # euclidean distance
    dist = [euclidean(noise_point, cm) for cm in cluster_means]

    # chebyshev distance
    dist = [chebyshev(noise_point, cm) for cm in cluster_means]

    # cityblock distance
    dist = [cityblock(noise_point, cm) for cm in cluster_means]

    # Helper Functions
    def calculate_cluster_means(X, labels):
        lbls = np.unique(labels)
        print "Cluster labels: {}".format(np.unique(lbls))
        cluster_means = [np.mean(X[labels == num, :], axis=0) for num in range(lbls[-1] + 1)]
        print "Cluster Means: {}".format(cluster_means)
        return cluster_means

    def print_3_distances(noise_point, cluster_means):
        # euclidean
        dist = [euclidean(noise_point, cm) for cm in cluster_means]
        print "Euclidean distance: {}".format(dist)
        # chebyshev
        dist = [chebyshev(noise_point, cm) for cm in cluster_means]
        print "Chebyshev distance: {}".format(dist)
        # cityblock
        dist = [cityblock(noise_point, cm) for cm in cluster_means]
        print "Cityblock (Manhattan) distance: {}".format(dist)

    def plot_the_clusters(X, dbscan_model, noise_point=None):
        labels = dbscan_model.labels_
        clusters = len(set(labels))
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12, 8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    if noise_point is not None:
        plt.plot(noise_point[0], noise_point[1], 'xr')

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)

    def do_yo_thang(X, dbscan_model, noise_point):
        cluster_means = calculate_cluster_means(X, dbscan_model.labels_)
        print_3_distances(noise_point, cluster_means)
        plot_the_clusters(X, dbscan_model, noise_point)

    X_ss = StandardScaler().fit_transform(X)
    db_ss = DBSCAN(eps=0.4, min_samples=3).fit(X_ss)
    noise_point = X_ss[30, :]
    do_yo_thang(X_ss, db_ss, noise_point)

    plt.savefig('task_3_dbscan.png')


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




