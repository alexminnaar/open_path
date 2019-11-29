import enum


class Measurement(object):
    '''

    '''
    def __init__(self, device, value, timestamp):
        '''

        :param value:
        :param timestamp:
        :param device:
        '''
        self.value = value
        self.timestamp = timestamp
        self.device = device


class Device(enum.Enum):
    '''

    '''
    radar = 1
    lidar = 2
