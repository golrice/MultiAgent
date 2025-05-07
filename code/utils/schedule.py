class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = duration
        self.step_count = 0

    def value(self) -> float:
        fraction = min(self.step_count / self.duration, 1.0)
        return self.start + fraction * (self.end - self.start)

    def step(self):
        self.step_count += 1

    def reset(self):
        self.step_count = 0
