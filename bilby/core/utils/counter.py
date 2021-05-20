import multiprocessing


class Counter(object):
    """
    General class to count number of times a function is Called, returns total
    number of function calls

    Parameters
    ==========
    initalval : int, 0
    number to start counting from
    """
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value
