import sys
sys.setrecursionlimit(10000)


class RecursiveDivider:
    def __init__(self):
        self.initial = True

    def list_divide(self, target: list):
        if len(target) <= 2:
            return tuple(t for t in target)
        else:
            half = int(len(target) / 2)
            if self.initial:
                a, b = target[:half], list(reversed(target[half:]))
                self.initial = False
            else:
                a, b = target[:half], target[half:]
            return [self.list_divide(a), self.list_divide(b)]

