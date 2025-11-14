import sys
import os

class LazyTee:
    def __init__(self, filename, mode="w"):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def _open_file(self):
        if self.file is None:
            os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
            self.file = open(self.filename, self.mode, buffering=1)

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()

        if message:
            self._open_file()
            self.file.write(message)
            self.file.flush()

    def flush(self):
        if self.file:
            self.file.flush()
        self.stdout.flush()

    def close(self):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
