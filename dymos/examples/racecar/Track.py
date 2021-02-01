import numpy as np


class Track:
    def __init__(self, segments):
        self.segments = segments
        self.cornerspeeds = np.zeros(len(segments), dtype=np.complex)

    def get_segment_type(self, num):
        return self.segments[num][0]

    def get_segment_length(self, num):
        return self.segments[num][1]

    def get_segment_radius(self, num):
        return self.segments[num][2]

    def get_corner_direction(self, num):
        return self.segments[num][3]

    def set_corner_speed(self, num, speed):
        self.cornerspeeds[num] = speed

    def get_corner_speed(self, num):
        return self.cornerspeeds[num]

    def get_total_length(self):
        length = 0
        for i in range(len(self.segments)):
            if self.segments[i][0] == 0:
                # straight
                length += self.get_segment_length(i)
            else:
                # corner
                length += self.get_segment_length(i) * self.get_segment_radius(i)
        return length
