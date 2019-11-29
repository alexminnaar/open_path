import enum

class Measurement(object):
    '''
    A measurement consisting of the measurement value, the device it was measured with and the time it was measured.
    '''
    def __init__(self, device, value, timestamp):
        '''
        Initialize measurement.
        :param value: Measurement value.
        :param timestamp: Time of measurement.
        :param device: Device used for measurement.
        '''
        self.value = value
        self.timestamp = timestamp
        self.device = device


class Device(enum.Enum):
    '''
    Either a radar or lidar measurement device.
    '''
    radar = 1
    lidar = 2
