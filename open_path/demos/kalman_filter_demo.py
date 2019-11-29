from open_path.tracking.kalman_filter import KalmanFilter
from open_path.tracking.measurement import Device, Measurement

data_location = "data/obj_pose-laser-radar-synthetic-input.txt"

kalman_filter = KalmanFilter()

with open(data_location, 'r') as f:
    for line in f:
        line_els = line.strip().split('\t')

        # extract measurement data
        if line_els[0] == 'R':
            device, rho, theta, ro_dot, ts, *_ = line_els
            measurement = Measurement(Device.radar, [float(rho), float(theta), float(ro_dot)], float(ts))

        elif line_els[0] == 'L':
            device, x, y, ts, *_ = line_els
            measurement = Measurement(Device.lidar, [float(x), float(y)], float(ts))

        else:
            raise ValueError("Unexpected device type: {}".format(line_els[0]))

        # complete one predict and update iteration of Kalman filter
        kalman_filter.filter_measurement(measurement)

print(kalman_filter.x)
print(kalman_filter.P)
