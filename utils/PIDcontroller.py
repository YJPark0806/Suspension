# utils/PIDcontroller.py

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, output_limits=(-float('inf'), float('inf'))):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def __call__(self, error, dt):
        self.integral += error * dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        low, high = self.output_limits
        return max(low, min(high, output))
