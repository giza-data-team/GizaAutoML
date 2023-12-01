from enum import Enum


class IntervalEnum(Enum):
    DAILY = 'D'
    MONTHLY = 'M'
    HOURLY = 'H'
    WEEKLY = 'W'
    MINUTE = 'T'
    SECOND = 'S'
    QUARTERLY = 'Q'
    YEARLY = 'Y'
