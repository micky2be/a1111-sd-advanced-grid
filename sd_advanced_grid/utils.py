# Python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

# Lib
import colorama as c
import numpy as np

# ################################# Constants ################################ #

FALSY = ["false", "no", "0"]
TRUTHY = ["true", "yes", "1"]
DEBUG_ICON = "\U0001F41B"  # 🐛
INFO_ICON = "\U0001F6C8"  # 🛈
WARN_ICON = "\U000026A0"  # ⚠
ERR_ICON = "\U000026D4"  # ⛔
CRIT_ICON = "\U000026A1"  # ⚡

# ############################# Utility Functions ############################ #


@dataclass(frozen=True)
class LogLevel:
    name: str
    severity: int
    prefix: str
    color: str


class Logger:
    DEBUG: ClassVar = LogLevel("debug", 10, "[D]", c.Fore.GREEN)
    INFO: ClassVar = LogLevel("info", 20, "[I]", c.Fore.BLUE)
    WARN: ClassVar = LogLevel("warning", 30, "[W]", c.Fore.MAGENTA)
    ERROR: ClassVar = LogLevel("error", 40, "[E]", c.Fore.RED)
    CRITICAL: ClassVar = LogLevel("critical", 50, "[C]", c.Fore.YELLOW + c.Style.BRIGHT)

    def __init__(self, level: LogLevel = INFO, *, color: bool = False):
        self.__min_sev = level.severity
        c.init(autoreset=False, strip=not color)

    def __print(self, level: LogLevel, msg: str, dataset):
        if level.severity >= self.__min_sev:
            print(f"\n{level.color}{level.prefix} {msg}")
            for sub_msg in dataset:
                print(f" * {sub_msg}")
            print(c.Style.RESET_ALL)

    def debug(self, msg: str, dataset: list | None = None):
        self.__print(Logger.DEBUG, msg, dataset or [])

    def info(self, msg, dataset: list | None = None):
        self.__print(Logger.INFO, msg, dataset or [])

    def warn(self, msg, dataset: list | None = None):
        self.__print(Logger.WARN, msg, dataset or [])

    def error(self, msg, dataset: list | None = None):
        self.__print(Logger.ERROR, msg, dataset or [])


# TODO: add cli args to config the logs
logger = Logger(Logger.DEBUG, color=True)


def clean_name(name: str):
    return str(name).lower().strip().replace(" ", "_").replace("[", "").replace("]", "")


def get_closest_from_list(name: str, choices: list[str]) -> str:
    if name in choices:
        return name

    found = sorted([item for item in choices if name in item], key=len)
    if found:
        return found[0]
    return ""


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(
    r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*"
)

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(
    r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*"
)


def parse_range_int(value_list: list[str]) -> list[int]:
    parsed_list = []

    for val in value_list:
        match = re_range.fullmatch(val)
        count = re_range_count.fullmatch(val)
        if match is not None:
            start = int(match.group(1))
            end = int(match.group(2)) + 1
            step = int(match.group(3)) if match.group(3) is not None else 1

            parsed_list += list(range(start, end, step))
        elif count is not None:
            start = int(count.group(1))
            end = int(count.group(2))
            num = int(count.group(3)) if count.group(3) is not None else 1
            parsed_list += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
        else:
            parsed_list.append(int(val))

    return parsed_list


def parse_range_float(value_list: list[str]) -> list[float]:
    parsed_list = []

    for val in value_list:
        match = re_range_float.fullmatch(val)
        count = re_range_count_float.fullmatch(val)
        if match is not None:
            start = float(match.group(1))
            end = float(match.group(2))
            step = float(match.group(3)) if match.group(3) is not None else 1
            parsed_list += np.arange(start, end + step, step).tolist()
        elif count is not None:
            start = float(count.group(1))
            end = float(count.group(2))
            num = int(count.group(3)) if count.group(3) is not None else 1
            parsed_list += np.linspace(start=start, stop=end, num=num).tolist()
        else:
            parsed_list += [round(float(val), 2)]

    return parsed_list
