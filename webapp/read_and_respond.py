import os
import sys
import logging
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import json
import requests
import urllib3

log = logging.getLogger(__name__)
urllib3.disable_warnings()
with open("appConfig.json") as f:
    try:
        app_config_json = json.load(f)
        log.info(f'app start up config appConfig.json app_config_json{app_config_json}: ')
    except Exception as e:
        log.error(f'error opening config file:{e}')
log.info(f'wsUrl:|{app_config_json["wsUrl"]} primarySiteRoute:|{app_config_json["primarySiteRoute"]}|')
primarySiteRoute = {'siteUrl': app_config_json["primarySiteRoute"], 'isLocked': 0, 'isActive': 1}
log.debug(f'primarySiteRoute:|{primarySiteRoute}|')
isScaffoldingTest = False


async def read_respond(self, event):
    if event != '':
        now = datetime.datetime.now()
        log.info(f'Received {event} @ {now}')

        log.debug(f'event["values"]:{event["values"]}')

        # This is where all your logic for responding to messages should go
        # use self.send() to respond to the webpage
        if event['name'] in ('state', 'heartbeat'):
            print('heartbeat')

        elif event['values'] == 'update site values':
            print('update')

        elif event['name'] == 'throw':
            await self.send('log', log)

    return


def getSiteRoutes():
    try:
        waReq = primarySiteRoute['siteUrl'] + 'getSiteConfigs'
        response = requests.get(waReq, verify=False)
        log.info(f'/GetArbs/getSiteConfigs:| {response.text}')
        routes = json.loads(response.text)
        # log.debug(f'routes:{routes}')
    except:
        log.error(f'/GetArbs/getSiteConfigs: error in request service may not be running')
        routes = ''
    return routes

