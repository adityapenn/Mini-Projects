import os
import json
import requests
import time
import re
import random

from flask import Flask, request, Response

application = Flask(__name__)

my_bot_name = "Aditya_Bot"
my_user_name = "apendyal"

slack_inbound_url = 'https://hooks.slack.com/services/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'


@application.route("/slack", methods = ["POST"])
def inbound():
    delay = random.uniform(0,10)
    time.sleep(delay)

    response = {"username": my_bot_name, "icon_url": "https://pbs.twimg.com/profile_images/751068553483214848/dm-oaLsx_400x400.jpg", "text": ''}

    channel = response.form.get('channel_name')
    username = request.form.get('user_name')
    text = request.form.get('text')
    inbound_message = username + " in " + channel + " says: " + text
    print("\n\nMessage:\n" + inbound_message)
    ntext = re.sub('<>', '', text)

        #Task1
    if username in [my_user_name, 'zac.wentzell'] and username != my_bot_name and re.findall("BOTS_RESPOND", ntext):

        response = {"username": my_bot_name, "icon_url": "https://pbs.twimg.com/profile_images/751068553483214848/dm-oaLsx_400x400.jpg", "text": ""}
        response['text']= 'Hello, my name is Aditya_bot. \n I belong to apendyal. \n I live at 54.190.57.163.'
        print(response['text'])
        requests.post(slack_inbound_url, json = response)


        #Task 2

    if username in [my_user_name, 'zac.wentzell'] and username != my_bot_name and re.findall(u'I_NEED_HELP_WITH_CODING', ntext):
        n2text = ntext[25:]
        stackap = 'api.stackexchange.com/2.2/search/advanced?order=desc&min=1&max=5&sort=relevance&title={}&q=&accepted=True&site=stackoverflow'.format(n2text)
        ans = requests.get(stackap)
        result = json.loads(ans.text)
        date = []
        link = []
        title = []

        for info in result['items'][:5]:
            date.append(info['creation_date'])
            link.append(info['link'])
            title.append(info['title'])
        c = ['#1F3B9A', '#C9D02F', '#BD1A18', '#EEF51E', '#C9D02F']
        atac = {
            "fallback": "My richly formatted text was not printed.",
            "color": c,
            "title": title,
            "link": link,
            "date": date
        }

        response = {
            "username": my_bot_name,
            "icon_url": "https://pbs.twimg.com/profile_images/751068553483214848/dm-oaLsx_400x400.jpg",
            "text": "Here are the answers I found:",
            "attachments": atac
        }

        requests.post(slack_inbound_url, json = response)

                # Task 3
    if username in [my_user_name, 'zac.wentzell'] and username != my_bot_name and re.findall(u'I_NEED_HELP_WITH_CODING', ntext) and re.findall(u'[', ntext):
        n2text = ntext[25:]
        n3text = re.sub("[<>]", "", str(n2text))
        stackap = 'api.stackexchange.com/2.2/search/advanced?order=desc&min=1&max=5&sort=relevance&q={}&title={}&accepted=True&site=stackoverflow'.format(
            n3text, n2text)
        ans = requests.get(stackap)
        result = json.loads(ans.text)
        date = []
        link = []
        title = []

        for info in result['items'][:5]:
            date.append(info['creation_date'])
            link.append(info['link'])
            title.append(info['title'])
        c = ['#1F3B9A', '#C9D02F', '#BD1A18', '#EEF51E', '#C9D02F']
        atac = {
            "fallback": "My richly formatted text was not printed.",
            "color": c,
            "title": title,
            "link": link,
            "date": date
        }

        response = {
            "username": my_bot_name,
            "icon_url": "https://pbs.twimg.com/profile_images/751068553483214848/dm-oaLsx_400x400.jpg",
            "text": "Here are the answers I found:",
            "attachments": atac
        }

        requests.post(slack_inbound_url, json=response)

            #Task 4
    if username in [my_user_name, 'zac.wentzell'] and username != my_bot_name and re.findall(u"WHAT'S_THE_WEATHER_LIKE_AT", ntext):
        zcode = re.sub("WHAT'S THE WEATHER LIKE AT", "", ntext)
        wapi = 'api.openweathermap.org/data/2.5/weather?zip={},us'.format(zcode)
        ans = requests.get(wapi)
        result = json.loads(ans.text)
        weather = result["weather"]["description"]
        humidity = result["main"]["humidity"]
        htemp = result["main"]["temp_max"]
        ltemp = result["main"]["temp_min"]
        wind = result["wind"]["speed"]
        text2 = "Here is how the weather is like at {}".format(zcode)
        winfo = [
            {
                "title": "The weather is",
                "value": weather,
                "short": True
            },
            {
                "title": "The humidity is",
                "value": humidity,
                "short": True
            },
            {
                "title": "The maximum temperature today is",
                "value": htemp,
                "short": True
            },
            {
                "title": "The minimum temperature today is",
                "value": ltemp,
                "short": True
            },
            {
                "title": "Today, the wind is at a speed of",
                "value": htemp,
                "short": True
            }
        ]
        atac2 = [
            {
                "color": c,
                "fields": winfo
            }
        ]
        response = {
            "username": my_bot_name,
            "icon_url": "https://pbs.twimg.com/profile_images/751068553483214848/dm-oaLsx_400x400.jpg",
            "text": text2,
            "attachments": atac2
        }

        requests.post(slack_inbound_url, json=response)
    if slack_inbound_url and response['text']:
        r = requests.post(slack_inbound_url, json = response)

    return Response(), 200

@application.route('/', methods = ['GET'])
def test():
    return Response('Your Flask app is running.')

if __name__ == "__main__":
    application.run(host = '0.0.0.0', port = 41953)
