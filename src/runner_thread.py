# runner_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
from src.util import ParameterException
import time


class RunnerThread(QThread):
    """
    Class to separate to a new thread the communications with the instruments.
    """
    get_dataset = pyqtSignal(dict)
    send_data = pyqtSignal(list, int)

    def __init__(self, sweep):
        """
        Initializes the object by taking in the parent sweep object, initializing the 
        plotter object, and calling the QThread initialization.
        
        Arguments:
            sweep - Object of type BaseSweep (or its children) that is controlling
                    this thread
        """
        QThread.__init__(self)

        self.sweep = sweep
        self.plotter = None
        self.datasaver = None
        self.dataset = None
        self.db_set = False
        self.kill_flag = False
        self.flush_flag = False
        self.runner = None

    def __del__(self):
        """
        Standard destructor.
        """
        self.wait()

    def add_plotter(self, plotter):
        """
        Adds the PlotterThread object, so the Runner knows where to send the new
        data for plotting.
        
        Arguments:
            plotter - PlotterThread object, should be same plotter created by parent sweep
        """
        self.plotter = plotter
        self.send_data.connect(self.plotter.add_data)

    def _set_parent(self, sweep):
        """
        Function to tell the runner who the parent is, if created independently.
        
        Arguments:
            sweep - Object of type BaseSweep, that Runner will be taking data for
        """
        self.sweep = sweep

    def run(self):
        """
        Function that is called when new thread is created. NOTE: start() is called
        externally to start the thread, but run() defines the behavior of the thread.
        Iterates the sweep, then hands the data to the plotter for live plotting.
        """
        # Check database status
        if self.sweep.save_data is True:
            self.runner = self.sweep.meas.run()
            self.datasaver = self.runner.__enter__()
            self.dataset = self.datasaver.dataset
            ds_dict = {}
            ds_dict['db'] = self.dataset.path_to_db
            ds_dict['run id'] = self.dataset.run_id
            ds_dict['exp name'] = self.dataset.exp_name
            ds_dict['sample name'] = self.dataset.sample_name
            self.get_dataset.emit(ds_dict)

        # print(f"called runner from thread: {QThread.currentThreadId()}")
        # Check if we are still running
        while self.kill_flag is False:
            t = time.monotonic()
            # print(f'kill flag = {str(self.kill_flag)}, is_running={self.sweep.is_running}')

            if self.sweep.is_running is True:
                # Get the new data
                try:
                    data = self.sweep.update_values()
                except ParameterException as e:
                    self.sweep.is_running = False
                    if e.set is True:
                        print(f"Could not set {e.p.label} to {e.sp}.", e.message)
                    else:
                        print(f"Could not get {e.p.label}.", e.message)
                    continue

                # Check if we've hit the end- update_values will return None
                if data is None:
                    continue

                # Send it to the plotter if we are going
                # Note: we check again if running, because we won't know if we are
                # done until we try to step the parameter once more
                if self.plotter is not None and self.sweep.plot_data is True:
                    self.send_data.emit(data, self.sweep.direction)

            # Smart sleep, by checking if the whole process has taken longer than
            # our sleep time
            sleep_time = self.sweep.inter_delay - (time.monotonic() - t)

            if sleep_time > 0:
                time.sleep(sleep_time)

            if self.flush_flag is True and self.sweep.save_data is True:
                self.datasaver.flush_data_to_database()
                self.flush_flag = False
            # print('at end of kill flag loop')

        if self.runner is not None:
            self.runner.__exit__(None, None, None)
