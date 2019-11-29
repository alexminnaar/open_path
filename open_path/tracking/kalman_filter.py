import numpy as np
from open_path.tracking.filter import Filter
import tensorflow as tf
from open_path.tracking.measurement import Measurement, Device


class KalmanFilter(Filter):
    '''
    Kalman filter implementation.
    '''

    def __init__(self):
        '''

        '''

        self.x = None
        self.P = None

        self.Q = None
        self.F = tf.convert_to_tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
            , dtype=tf.float32)

        self.P = tf.convert_to_tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]]
            , dtype=tf.float32)

        self.R_radar = tf.convert_to_tensor(
            [[0.09, 0, 0], [0, 0.0009, 0], [0, 0, 0.09]]
            , dtype=tf.float32
        )

        self.R_lidar = tf.convert_to_tensor(
            [[0.0225, 0], [0, 0.0225]]
            , dtype=tf.float32
        )

        self.H_lidar = tf.convert_to_tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0]]
            , dtype=tf.float32
        )

        self.acc_noise_x = 9
        self.acc_noise_y = 9

        self.initialized = False
        self.prev_timestamp = None

    def predict(self):
        '''

        :return:
        '''
        self.x = tf.matmul(self.F, self.x)
        self.P = tf.matmul(tf.matmul(self.F, self.P), tf.transpose(self.F)) + self.Q

    def lidar_measurement_prediction(self):
        '''

        :return:
        '''
        z_pred = tf.matmul(self.H_lidar, self.x)
        return z_pred

    def radar_measurement_prediction(self):
        '''

        :return:
        '''
        px, py, vx, vy = self.x.numpy()

        # convert state space into measurement space
        h_np = [np.sqrt(px[0] * px[0] + py[0] * py[0]), np.arctan2(py[0], px[0]),
                (px[0] * vx[0] + py[0] * vy[0]) / np.sqrt(px[0] * px[0] + py[0] * py[0])]

        return tf.expand_dims(tf.convert_to_tensor(h_np, dtype=tf.float32), 1)

    def update(self, z_):
        '''

        :param z:
        :return:
        '''

        # convert numpy measurement to tensor
        z = tf.expand_dims(tf.convert_to_tensor(z_.value), 1)

        # compute measurement prediction based on measurement type
        if z_.device == Device.radar:
            z_pred = self.radar_measurement_prediction()
            H = self.compute_jacobian()
            R = self.R_radar
            y = z - z_pred

            y_np = y.numpy()

            while y_np[1][0] > np.pi:
                y_np[1][0] -= 2 * np.pi

            while y_np[1][0] < -np.pi:
                y_np[1][0] += 2 * np.pi

            y = tf.convert_to_tensor(y_np, dtype=tf.float32)

        elif z_.device == Device.lidar:
            z_pred = self.lidar_measurement_prediction()
            H = self.H_lidar
            R = self.R_lidar
            y = z - z_pred

        else:
            raise ValueError('Unexpected measurement device: {}'.format(z_.device))

        S = tf.matmul(tf.matmul(H, self.P), tf.transpose(H)) + R
        K = tf.matmul(tf.matmul(self.P, tf.transpose(H)), tf.linalg.inv(S))

        self.x = self.x + tf.matmul(K, y)
        x_len = self.x.get_shape()[0]
        I = tf.eye(x_len, x_len)
        self.P = tf.matmul((I - tf.matmul(K, H)), self.P)

    def compute_jacobian(self):
        '''

        :return:
        '''
        px, py, vx, vy = self.x.numpy()

        d1 = px[0] * px[0] + py[0] * py[0]
        d2 = np.sqrt(d1)
        d3 = d1 * d2

        if np.abs(d1) < 0.0001:
            raise ValueError("Division by zero in Jacobian calculation")

        H = tf.convert_to_tensor(
            [[px[0] / d2, py[0] / d2, 0, 0], [-(py[0] / d1), px[0] / d1, 0, 0],
             [py[0] * (vx[0] * py[0] - vy[0] * px[0]) / d3, px[0] * (px[0] * vy[0] - py[0] * vx[0]) / d3, px[0] / d2,
              py[0] / d2]]
            , dtype=tf.float32
        )

        return H

    def filter_measurement(self, z):
        '''

        :param z:
        :return:
        '''

        if not self.initialized:

            # initialize state vector
            if z.device == Device.radar:
                self.x = tf.convert_to_tensor(
                    [[z.value[0] * np.cos(z.value[1])], [z.value[0] * np.sin(z.value[1])], [0], [0]], dtype=tf.float32)
            elif z.device == Device.lidar:
                self.x = tf.convert_to_tensor([[z.value[0]], [z.value[1]], [0], [0]], dtype=tf.float32)
            else:
                raise ValueError('Unexpected measurement device: {}'.format(z.device))

            self.initialized = True
            self.prev_timestamp = z.timestamp

        else:

            # time difference in seconds
            dt = (z.timestamp - self.prev_timestamp) / 1000000.0
            self.prev_timestamp = z.timestamp

            # update F matrix with new time difference
            F_np = self.F.numpy()
            F_np[0][2] = dt
            F_np[1][3] = dt
            self.F = tf.convert_to_tensor(F_np, dtype=tf.float32)

            self.Q = tf.convert_to_tensor(
                [[(dt ** 4) / 4 * self.acc_noise_x, 0, (dt ** 3) / 2 * self.acc_noise_x, 0],
                 [0, (dt ** 4) / 4 * self.acc_noise_y, 0, (dt ** 3) / 2 * self.acc_noise_y],
                 [(dt ** 3) / 2 * self.acc_noise_x, 0, (dt ** 2) * self.acc_noise_x, 0],
                 [0, (dt ** 3) / 2 * self.acc_noise_y, 0, (dt ** 2) * self.acc_noise_y]]
                , dtype=tf.float32
            )

            # predict then update based on prediction
            self.predict()
            self.update(z)
