import logging
import sys

import data_balancer
import logger_utils
from file_utils import read_data, save_data
from settings import TrainerSetting, get_setting, SettingType

log = logging.getLogger("trainer")


def execute_trainer(setting: TrainerSetting) -> None:
    data = read_data(setting)
    if setting.use_data_balancing:
        data, y = data_balancer.execute(data, setting.data_balancing_setting)

    save_data(data, setting.output_data_path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide trainer settings path. Exit...")
    elif len(sys.argv) == 2:
        trainer_settings = get_setting(str(sys.argv[1]), SettingType.trainer)
        logger_utils.init_logging(trainer_settings.log_path + "\\" + trainer_settings.name)
        execute_trainer(trainer_settings)
    else:
        print("Unknown number of arguments! Exit...")
