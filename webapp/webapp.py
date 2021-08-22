import os
import sys
import threading
import logging
import datetime
import time

from waitress import serve
from flask import Flask, abort, current_app, request, url_for, jsonify
from flask_cors import CORS
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from config import configuration
from flask_restful import Resource, Api
from functools import wraps
import uuid
import urllib3

config = configuration()
host = config['web']['host']
port = config['web']['port']
urllib3.disable_warnings()
log = logging.getLogger(__name__)
accountSeq = 0

tasks = {}

app = Flask(__name__)
api = Api(app)
app.config.from_object(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/')
def emptyrout():
    return "root route path: "


@app.route('/ping')
def ping():
    return 'ring-a-ding-ding'


@app.before_first_request
def before_first_request():
    """Start a background thread that cleans up old tasks and performs other operations in this case every 60 seconds."""

    def clean_old_tasks():
        """
        This function cleans up old tasks from our in-memory data structure.
        """
        global tasks
        while True:
            # Only keep tasks that are running or that finished less than 5
            # minutes ago.
            refresh_time_out = datetime.datetime.timestamp(datetime.datetime.now()) - 5 * 60
            tasks = {task_id: task for task_id, task in tasks.items()
                     if 'completion_timestamp' not in task or task['completion_timestamp'] > refresh_time_out}

            time.sleep(120)
            log.info(f"cleared tasks completed more than 5 minutes ago")

    if not current_app.config['TESTING']:
        thread = threading.Thread(target=clean_old_tasks)
        thread.start()


class GetTaskStatus(Resource):
    def get(self, task_id):
        """
        Return status about an asynchronous task. If this request returns a 202
        status code, it means that task hasn't finished yet. Else, the response
        from the task is returned.
        """
        task = tasks.get(task_id)
        if task is None:
            abort(404)
        if 'return_value' not in task:
            return '', 202, {'Location': url_for('gettaskstatus', task_id=task_id)}
        return task['return_value']


if __name__ == '__main__':
    # app.run(ssl_context='adhoc', host=tIp, port=tPort)
    serve(app, host=host, port=port, url_scheme='http')
