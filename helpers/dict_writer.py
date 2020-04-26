#!/usr/bin/env python

import csv
from multiprocessing import Lock


class SyncWriterWrapper():
    def __init__(self, path):
        self.lock = Lock()
        self.p = open(path, 'w')

    def writerow(self, row):
        self.lock.acquire()
        try:
            print('Writing row: ', row)
            print(row, file=self.p, flush=True)
        finally:
            self.lock.release()

    def writeheader(self):
        self.writer.writeheader()

    def close(self):
        self.p.close()
