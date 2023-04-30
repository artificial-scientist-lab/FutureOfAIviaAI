import datetime


class Timer(object):
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        print(f'{self.name} time elapsed: {(datetime.datetime.now() - self.start).seconds} seconds...')
