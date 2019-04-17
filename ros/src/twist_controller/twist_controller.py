
import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

MIN_SPEED   = 0.1   # m/s
GAS_DENSITY = 2.858
ONE_MPH     = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)

        # Use a PID controller for the throttle.
        # Parameters determined experimentally -> provided by Udacity.
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0.     # Min. throttle value
        mx = 0.2    # Max. throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # Use a low pass filter to filter high-frequency noise from velocity input.
        tau = 0.5   # 1 / (2 * pi * tau) == cutoff frequency
        ts  = 0.02  # Sample time (50 Hz)
        self.velocity_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass   = vehicle_mass
        self.fuel_capacity  = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit    = decel_limit
        self.accel_limit    = accel_limit
        self.wheel_radius   = wheel_radius

        self.last_velocity  = 0
        self.last_time      = rospy.get_time()

    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steering

        # If Drive-By-Wire is not enabled, i.e., the human driver is in control, then reset
        # the PID controller (so we are not accululating integral error while the car is
        # idling) and just return zeros for throttle, brake, steering.
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        rospy.logwarn('Target linear velocity : {}'.format(linear_velocity))
        rospy.logwarn('Target angular velocity: {}'.format(angular_velocity))
        rospy.logwarn('Current velocity : {}'.format(current_velocity))

        # Apply the low pass filter to the current velocity.
        current_velocity = self.velocity_lpf.filt(current_velocity)
        rospy.logwarn('Filtered velocity: {}'.format(current_velocity))

        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        # The error is the difference between the target velocity and the current velocity.
        velocity_error = linear_velocity - current_velocity

        # The sample time is the elapsed time since the last call to this controller.
        current_time = rospy.get_time()
        sample_time  = current_time - self.last_time
        self.last_time = current_time

        # Get the throttle value from the PID controller.
        # Throttle is in the range [0., 1.]
        throttle = self.throttle_controller.step(velocity_error, sample_time)
        brake    = 0.

        # Apply braking as necessary: the brake value is expressed as torque measured in Nm.

        # If the target velocity is 0 and our curent velocity is essentially 0,
        # then assume we're stopped at a traffic light.
        if linear_velocity == 0. and current_velocity < MIN_SPEED:
            throttle = 0.
            brake    = 700. # Nm - hold the car in place if we are stopped at a traffic light
                            # Acceleration == 1 m/s^2

        # Otherwise if the throttle is essentially 0 and the target velocity is
        # less than the current velocity, apply braking.
        elif throttle < 0.1 and velocity_error < 0:
            throttle = 0.
            decel    = max(velocity_error, self.decel_limit)
            # torque: vehicle mass in kg, wheel radius in m
            # brake    = abs(decel) * (self.vehicle_mass + (self.fuel_capacity * GAS_DENSITY)) * self.wheel_radius
            brake    = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
