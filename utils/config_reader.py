from configparser import RawConfigParser
from pathlib import Path
from typing import List


class ConfigReader:
    """
    配置读取器
    """

    def __init__(self, config_path: str):
        """
        构造函数
        :param config_path: 配置文件相对于工程根目录的位置
        """
        self.configparser = RawConfigParser()
        abs_path = Path(__file__).parent.parent.joinpath(config_path).resolve()
        self.configparser.read(abs_path)

    def read_config(self, section: str, item_name: str) -> str:
        """
        读取单个配置
        :param section: 配置区域
        :param item_name: 配置属性名
        :return:
        """
        section = self.configparser[section]
        return section[item_name]

    def read_configs(self, section: str, item_names: List[str]) -> List[str]:
        """
        读取指定section下面的指定属性值到列表
        :param section: 指定section
        :param item_names: 属性名列表
        :return:
        """
        section = self.configparser[section]
        return [section[item] for item in item_names]
