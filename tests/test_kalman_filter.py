from open_path.tracking.measurement import Device, Measurement
from open_path.tracking.kalman_filter import KalmanFilter
import tensorflow as tf
import unittest


class TestKalmanFilter(unittest.TestCase):
    kalman_filter = KalmanFilter()

    #TODO: test P as well
    def test_filter(self):
        test_meas_1 = Measurement(Device.radar, [15.46745, 2.394427, -4.449697], 1477010467150000)
        test_meas_2 = Measurement(Device.lidar, [-10.60086, 10.68225], 1477010467200000)
        test_meas_3 = Measurement(Device.radar, [15.19847, 2.347791, -3.728088], 1477010467250000)
        test_meas_4 = Measurement(Device.lidar, [-10.41423, 10.87465], 1477010467300000)

        self.kalman_filter.filter_measurement(test_meas_1)
        self.kalman_filter.filter_measurement(test_meas_2)
        self.kalman_filter.filter_measurement(test_meas_3)
        self.kalman_filter.filter_measurement(test_meas_4)

        correct_x = tf.convert_to_tensor(
            [[-10.3064],
             [10.754204],
             [4.451833],
             [-0.5411225]],
            dtype=tf.float32
        )

        self.assertTrue(tf.reduce_all(tf.equal(self.kalman_filter.x, correct_x)))


if __name__ == '__main__':
    unittest.main()
