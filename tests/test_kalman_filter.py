from open_path.tracking.kalman_filter import KalmanFilter
import unittest
import tensorflow as tf


class TestKalmanFilter(unittest.TestCase):
    x_ = [[0], [0]]
    P_ = [[1000, 0], [1000, 0]]
    F_ = [[1, 1], [0, 1]]
    H_ = [[1, 0]]
    R_ = [[1]]
    Q_ = [[0], [0]]


    kalman_filter = KalmanFilter(x_, P_, F_, H_, R_, Q_)

    def test_tensor_slicing(self):
        x_ = tf.convert_to_tensor(self.x_, dtype=tf.float32)
        print(x_)
        print(x_.numpy())

    def test_predict(self):
        self.kalman_filter.predict()

    def test_update(self):
        z = [[1]]
        self.kalman_filter.update(z)


if __name__ == '__main__':
    unittest.main()
