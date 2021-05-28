# -*- coding: utf-8 -*-

import atexit
import json
import logging
import os
import sys
from pathlib import WindowsPath

import werkzeug.exceptions
from flask import Flask, request, Response, jsonify

# add path to sources for production
sys.path.append(str(WindowsPath(os.path.dirname(os.path.abspath(__file__))).parent))

from model_initializer import init_predictor, ServiceParameterPredictor
from settings import get_setting, SettingType, PreprocessorSetting, ServiceSetting
from utils import logger_utils

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
log = logging.getLogger("ml_service")

SERVICE_CONFIG = "ML_SERVICE_SETTINGS"
service_settings: ServiceSetting = None
preprocessor_settings: PreprocessorSetting = None
predictor: ServiceParameterPredictor = None

app = Flask(__name__)


def on_exit_app():
    log.info("Application shout down!")


@app.errorhandler(werkzeug.exceptions.NotFound)
def handle_invalid_usage(error) -> Response:
    response = jsonify(error.description)
    response.status_code = error.code
    log.error("Got exception: " + str(error), exc_info=True)
    return response


@app.errorhandler(Exception)
def handle_invalid_usage(error) -> Response:
    log.error("Got exception: " + str(error), exc_info=True)
    return Response(response=str(error), status=500)


@app.before_request
def log_request_info() -> None:
    if request.json:
        app.logger.info('Body: %s', request.get_json())


@app.after_request
def log_response_info(response: Response) -> Response:
    if response.json:
        log.info(f'Response: \n {response.get_json()}')
    return response


@app.get('/health')
def check_health() -> Response:
    return jsonify('status: UP')


@app.get('/show_model')
def show_model() -> Response:
    if predictor.model is not None:
        summary = []
        predictor.model.summary(print_fn=log.info)
        predictor.model.summary(print_fn=lambda x: summary.append(x))
        return Response(response='\n'.join(summary), status=200, mimetype='text/plain')
    else:
        return Response(response='Model not loaded!', status=200)


@app.post('/predict')
def predict() -> Response:
    if not request.json or 'text' not in request.json:
        return Response(response=f'Request does not contain json or "text" attribute', status=500)
    if request.json['text'] == "":
        return Response(json.dumps({"predictions": None}), status=200, mimetype='application/json')
    text = predictor.preprocess_text(request.json['text'], preprocessor_settings)
    log.info(f'Preprocessed text: {text}')
    predictions = predictor.get_prediction(text, service_settings.top_n_predictions)
    if predictions.shape[0] == 0:
        return Response(json.dumps({"predictions": None}), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"predictions": predictions.to_dict(orient="records")}), status=200,
                        mimetype='application/json')


def init_service(service_setting: ServiceSetting, preprocessor_setting: PreprocessorSetting) -> None:
    global predictor
    log.info("==> Start service initialization")
    predictor = init_predictor(service_setting, preprocessor_setting)
    log.info("==> Service initialized")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        setting_path = str(sys.argv[1])
    elif os.getenv(SERVICE_CONFIG) != "":
        setting_path = os.getenv(SERVICE_CONFIG)
    else:
        raise ValueError(f'Can not find service setting')
    service_settings: ServiceSetting = get_setting(setting_path, SettingType.service)
    preprocessor_settings: PreprocessorSetting = get_setting(service_settings.preprocessor_settings_path,
                                                             SettingType.cleaner)
    logger_utils.init_logging(service_settings.log_path + "\\" + service_settings.name)
    log.info(f'Using service:\n{json.dumps(service_settings.__dict__, default=lambda x: x.__dict__)}')
    log.info(f'Using preprocessor:\n{json.dumps(preprocessor_settings.__dict__, default=lambda x: x.__dict__)}')
    init_service(service_settings, preprocessor_settings)
    app.run(debug=False, threaded=True)
    atexit.register(on_exit_app)
    log.info("Flask server started!")
