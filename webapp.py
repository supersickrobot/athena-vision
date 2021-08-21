import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from logging.handlers import QueueHandler
import webapp.read_and_respond
from webapp.ws import WSHandler
from aiohttp import web
import asyncio
from webapp.logger import get_logger
import janus
import json

log = get_logger("webapp/logs/PiVue.log")

isScaffoldingTest = False
# retrieve app configs
cwd = os.getcwd()
log.debug(f'cwd:{cwd}')
with open("appConfig.json") as f:
    try:
        app_config_json = json.load(f)
        log.info(f'app start up config appConfig.json app_config_json{app_config_json}: ')
    except Exception as e:
        log.error(f'error opening config file:{e}')
log.info(f'wsUrl:|{app_config_json["wsUrl"]}')

async def main():
    # logging.basicConfig(level=logging.DEBUG, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')

    ws_handler = WSHandler()
    sub_app = web.Application()
    sub_app['ws'] = ws_handler

    ws_app = web.Application()
    ws_app.add_routes([web.get('/', ws_handler.handler)])

    app = web.Application()
    app.add_subapp('/api/v1', sub_app)
    app.add_subapp('/ws', ws_app)
    await asyncio.gather(
        web._run_app(app, host=app_config_json["wsUrl"], port=app_config_json["wsUrlPort"]),  # using appConfig.json in root
        heartbeat(ws_handler),
        log_display(ws_handler)
        #this is your entry point for new async loops
    )

async def log_display(wshandler):
    log_queue = janus.Queue()
    q_log = QueueHandler(log_queue.sync_q)
    # q_log.setFormatter(logging.Formatter() todo this is where you can add formatting for your logs
    logger = logging.getLogger()
    logger.addHandler(q_log)
    while True:
        try:
            new_log = await log_queue.async_q.get()
            await wshandler.send('log', {'msg': new_log.msg})
        except asyncio.CancelledError:
            log.info('Stopping log display')
            break

async def heartbeat(wshandler):

    while True:
        log.debug('start heartbeat')
        # await wshandler.send('heartbeat', {}) #  to test message panel
        if not isScaffoldingTest:
            await wshandler.send('state', None)
            arbData = await webapp.read_and_respond.getArbOpData()
            await wshandler.send('arbs', arbData)
            arbCount = len(arbData)
            await wshandler.send('arbCount', arbCount)
            openCompetitionsData = await webapp.read_and_respond.getOpenCompetitionsData()
            await wshandler.send('openCompetitions', openCompetitionsData)
            OpenBS1Data = await webapp.read_and_respond.getOpenBS1()
            await wshandler.send('OpenBS1', OpenBS1Data)
            OpenBS2Data = await webapp.read_and_respond.getOpenBS2()
            await wshandler.send('OpenBS2', OpenBS2Data)
            StrategyActualData = await webapp.read_and_respond.getStrategyActual()
            await wshandler.send('StrategyActual', StrategyActualData)
            CashDetailData = await webapp.read_and_respond.getCashDetails()
            await wshandler.send('CashDetail', CashDetailData)
            CashSummaryData = await webapp.read_and_respond.getCashSummary()
            await wshandler.send('CashSummary', CashSummaryData)
            PnLSummaryData = await webapp.read_and_respond.getPnLSummary()
            await wshandler.send('PnLSummary', PnLSummaryData)
            graphs = await webapp.read_and_respond.getPlotData()
            await wshandler.send('graphs', graphs)

        await asyncio.sleep(60)


if __name__ == '__main__':
    asyncio.run(main())