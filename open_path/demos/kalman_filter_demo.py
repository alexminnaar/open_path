from open_path.tracking.measurement import Device, Measurement
from open_path.tracking.kalman_filter import KalmanFilter

data_location = "data/obj_pose-laser-radar-synthetic-input.txt"

kalman_filter = KalmanFilter()

# each line of data file is either a radar or lidar measurement - feed them into the Kalman filter to update estimate.
with open(data_location, 'r') as f:
    for line in f:
        line_els = line.strip().split('\t')

        # extract measurement
        if line_els[0] == 'R':
            device, rho, theta, ro_dot, ts, *_ = line_els
            measurement = Measurement(Device.radar, [float(rho), float(theta), float(ro_dot)], float(ts))
            print(measurement.device,measurement.value, measurement.timestamp)
        elif line_els[0] == 'L':
            device, x, y, ts, *_ = line_els
            measurement = Measurement(Device.lidar, [float(x), float(y)], float(ts))
            print(measurement.device,measurement.value, measurement.timestamp)

        else:
            raise ValueError("Unexpected device type: {}".format(line_els[0]))

        # complete a predict and update iteration of Kalman filter
        kalman_filter.filter_measurement(measurement)

print(kalman_filter.x)
print(kalman_filter.P)
