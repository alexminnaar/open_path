from open_path.tracking.filter import Filter
import tensorflow as tf


class KalmanFilter(Filter):
    '''
    Kalman filter implementation.
    '''

    def __init__(self, x_, P_, F_, H_, R_, Q_):
        '''

        :param x_:
        :param P_:
        :param F_:
        :param H_:
        :param R_:
        :param Q_:
        '''

        self.x_ = tf.convert_to_tensor(x_, dtype=tf.float32)
        self.P_ = tf.convert_to_tensor(P_, dtype=tf.float32)
        self.F_ = tf.convert_to_tensor(F_, dtype=tf.float32)
        self.H_ = tf.convert_to_tensor(H_, dtype=tf.float32)
        self.R_ = tf.convert_to_tensor(R_, dtype=tf.float32)
        self.Q_ = tf.convert_to_tensor(Q_, dtype=tf.float32)

    def predict(self):
        '''

        :return:
        '''
        self.x_ = tf.matmul(self.F_, self.x_)
        self.P_ = tf.matmul(tf.matmul(self.F_, self.P_), tf.transpose(self.F_)) + self.Q_

    def update(self, z):
        '''

        :param z:
        :return:
        '''
        z_pred = tf.matmul(self.H_, self.x_)
        y = z - z_pred
        S = tf.matmul(tf.matmul(self.H_, self.P_), tf.transpose(self.H_)) + self.R_
        K = tf.matmul(tf.matmul(self.P_, tf.transpose(self.H_)), tf.linalg.inv(S))

        self.x_ = self.x_ + tf.matmul(K, y)
        I = tf.eye(tf.size(self.x_), tf.size(self.x_))
        self.P_ = tf.matmul((I - tf.matmul(K, self.H_)), self.P_)
