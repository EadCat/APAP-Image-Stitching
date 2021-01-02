from option import Options
from thread import Thread

if __name__ == '__main__':
    opt = Options().parse()
    thread = Thread(opt)
    thread.thread_choice()
