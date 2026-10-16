from option import Options
from thread import Thread
from stitching.apap import Apap


class Tuner(Thread):
    """
    recommend one (sigma, gamma) for each stitching case.
    A case of N images has N-1 stitching stages, and the held-out errors of all stages are summed.
    """
    def process(self, imgdir, mask):
        self.param_tables = []
        super().process(imgdir, mask)
        if not self.param_tables:
            print(f'{self.n} case: no stage to evaluate.')
            return
        sse = sum(table[0] for table in self.param_tables)
        count = sum(table[1] for table in self.param_tables)
        sigma, gamma = Apap.param_select(sse, count)
        print(f'{self.n} case ({len(self.param_tables)} stages) -> --sigma {sigma} --gamma {gamma}')


if __name__ == '__main__':
    opt = Options().parse()
    opt.auto_param = True
    Tuner(opt).thread_choice()
