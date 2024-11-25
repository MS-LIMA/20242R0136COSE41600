import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
    
    def reset(self):
        self.start_time = time.time()
        
    def get_passed_time(self, reset=True):
        current_time = time.time()
        passed_time = round(current_time - self.start_time, 3)
        if reset:
            self.start_time = current_time
        return passed_time
        