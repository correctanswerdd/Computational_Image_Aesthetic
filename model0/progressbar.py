import sys
import time


class ProgressBar:

    def __init__(self, max, nstep=100, char_done='=', char_todo=' ', mode='percentage', info_done='done'):
        assert char_done != char_todo and len(char_done) == 1 and len(char_todo) == 1
        self.__max = max
        self.__nstep = nstep
        self.__char_done = char_done
        self.__char_todo = char_todo
        self.__mode = mode
        self.__info_done = info_done
        self.__running = False
        self.__start_time = self.__current_time = 0
        self.__current = 0

    @property
    def max(self):
        return self.__max

    @max.setter
    def max(self, max):
        if self.running:
            return
        self.__max = max

    @property
    def current(self):
        return self.__current

    @current.setter
    def current(self, current):
        self.__current = max(min(current, self.max), 0)

    @property
    def nstep(self):
        return self.__nstep

    @nstep.setter
    def nstep(self, nstep):
        if self.running:
            return
        self.__nstep = nstep

    @property
    def running(self):
        return self.__running

    @running.setter
    def running(self, running: bool):
        self.__running = running

    @property
    def start_time(self):
        return self.__start_time

    @property
    def current_time(self):
        return self.__current_time

    @property
    def char_done(self):
        return self.__char_done

    @char_done.setter
    def char_done(self, char_done):
        if char_done != self.char_todo and len(char_done) == 1:
            self.__char_done = char_done

    @property
    def info_done(self):
        return self.__info_done

    @info_done.setter
    def info_done(self, info_done):
        self.__info_done = info_done

    @property
    def char_todo(self):
        return self.__char_todo

    @char_todo.setter
    def char_todo(self, char_todo):
        if char_todo != self.char_done and len(char_todo) == 1:
            self.__char_todo = char_todo

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        if mode not in ['percentage', 'fraction']:
            mode = 'percentage'
        self.__mode = mode

    def show(self):
        if not self.running:
            return
        self.__current_time = time.time()
        nprogress = int(self.current / self.max * self.nstep)

        if self.mode == 'percentage':
            info = '%.2f' % (self.current / self.max * 100) + '%'
        else:
            info = str(round(self.current, 2)) + '/' + str(round(self.max, 2))

        progress_bar = '[' + self.char_done * nprogress + self.char_todo * (self.nstep - nprogress) + '] ' + info
        sys.stdout.write('\r' + progress_bar)
        sys.stdout.flush()

    def show_progress(self, progress):
        self.current = progress
        self.show()

    def start(self):
        if self.running:
            return
        self.running = True
        self.__start_time = time.time()
        self.show()

    def end(self):
        if not self.running:
            return
        self.__current_time = time.time()
        print('')
        print('done, used', round(self.current_time - self.start_time, 2), 's')
        self.__start_time = 0
        self.__current_time = 0
        self.current = 0
        self.running = False
