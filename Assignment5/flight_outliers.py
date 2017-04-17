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




