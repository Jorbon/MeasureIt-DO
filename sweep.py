import io
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import qcodes as qc
from qcodes.dataset.measurements import Measurement
from IPython import display
from PyQt5.QtCore import QThread, pyqtSignal

def _autorange_srs(srs, max_changes=1):
    def autorange_once():
        r = srs.R.get()
        sens = srs.sensitivity.get()
        if r > 0.9 * sens:
            return srs.increment_sensitivity()
        elif r < 0.1 * sens:
            return srs.decrement_sensitivity()
        return False
    sets = 0
    while autorange_once() and sets < max_changes:
        sets += 1
        time.sleep(10*srs.time_constant.get())

class Sweep1D(object):
    """
    Class to control sweeping along 1 parameter, while tracking multiple other parameters.
    It has functionality to live plot, or not, and one can create their own figures (e.g. in a GUI)
    or have the class create its own matplotlib figues. Follows QCoDeS data-taking methods.
    Adapted from Joe Finney's code.
    
    SR830s are not currently implemented.
    """
    
    def __init__(self, set_param, start, stop, step, freq, meas=None, plot=False, auto_figs=False):
        """
        Initializes the Sweep object. Takes in the parameter to be swept (set_param), the 
        value to start and stop sweeping at (start/stop, respectively), the step spacing (step),
        and the frequency of measurements. Can turn plotting off through 'plot', also tells 
        system whether to create it's own plots or use given ones through 'auto_figs'.
        """
        # Save our input variables
        self.set_param = set_param
        self.start = start
        self.stop = stop
        self.step = step
        self.inter_delay = 1/freq
        self.t0 = time.monotonic()
        self.setpoint = self.start - self.step
        
        d = (stop-start)/step*self.inter_delay
        h, m, s = int(d/3600), int(d/60) % 60, int(d) % 60
        print(f'Minimum duration: {h}h {m}m {s}s')
        
        # Either mark or save the measurement object
        if meas is None:
            self.meas = self._create_measurement(self.set_param)
        else:
            self.meas = meas
            
        # Saves our plotting flags
        self.plot = plot
        self.auto_figs = auto_figs
        # Sets a flag to ensure that the figures have been created before trying to plot
        self.figs_set = False
        self._sr830s = []
        self._params = []
    
    def follow_param(self, p):
        """
        This function takes in a QCoDeS Parameter p, and tracks it through each sweep.
        """
        self._params.append(p)

    def follow_sr830(self, l, name, gain=1.0):
        """
        This function adds an SR830, but (as of now) does not do anything with it.
        """
        self._sr830s.append((l, name, gain))

    def _create_measurement(self, *set_params):
        """
        Creates a QCoDeS Measurement object. This controls the saving of data by registering
        QCoDeS Parameter objects, which this function does. Registers all 'sweeping' parameters
        (set_params), and all 'tracked' parameters, 'self._params'. Returns the measurement
        object.
        """
        self.meas = Measurement()
        for p in set_params:
            self.meas.register_parameter(p)
        self.meas.register_custom_parameter('time', label='Time', unit='s')
        for p in self._params:
            self.meas.register_parameter(p, setpoints=(*set_params, 'time',))
        for l, _, _ in self._sr830s:
            self.meas.register_parameter(l.X, setpoints=(*set_params, 'time',))
            self.meas.register_parameter(l.Y, setpoints=(*set_params, 'time',))
            
        return self.meas
        
    def create_figs(self):
        """
        Creates default figures for each of the parameters. Plots them in a new, separate window.
        """
        self.fig = plt.figure(figsize=(4*(2 + len(self._params) + len(self._sr830s)),4))
        self.grid = plt.GridSpec(4, 1 + len(self._params) + len(self._sr830s), hspace=0)
        self.setax = self.fig.add_subplot(self.grid[:, 0])
        # First, create a plot of the sweeping parameters value against time
        self.setax.set_xlabel('Time (s)')
        self.setax.set_ylabel(f'{self.set_param.label} ({self.set_param.unit})')
        self.setaxline = self.setax.plot([], [])[0]
        
        self.plines = []
        self.axes = []
        # Now create a plot for every tracked parameter as a function of sweeping parameter
        for i, p in enumerate(self._params):
            self.axes.append(self.fig.add_subplot(self.grid[:, 1 + i]))
            self.axes[i].set_xlabel(f'{self.set_param.label} ({self.set_param.unit})')
            self.axes[i].set_ylabel(f'{p.label} ({p.unit})')
            self.plines.append(self.axes[i].plot([], [])[0])
            
        self.figs_set = True
            
    def set_figs(self, fig, setax, axes):
        """
        Give a figure and plots for both the sweeping parameter and the tracked parameters
        for the program to update. fig is of type matplotlib Figure, setax is a (sub)plot, 
        and axes is an array of subplots.
        """
        self.figs_set = True
        self.fig = fig
        self.setax = setax
        self.axes = axes
    
        # Initializes sweeping plot
        self.setax.set_xlabel('Time (s)')
        self.setax.set_ylabel(f'{self.set_param.label} ({self.set_param.unit})')
        self.setaxline = setax.plot([], [])[0]
        
        self.plines = []
        # Initializes tracking plots
        for i, p in enumerate(self._params):
            self.axes[i].set_xlabel(f'{self.set_param.label} ({self.set_param.unit})')
            self.axes[i].set_ylabel(f'{p.label} ({p.unit})')

            self.plines.append(self.axes[i].plot([], [])[0])
    
    def autorun(self):
        """
        Run a sweep through this class. Makes call to create_figs if needed.
        Calls self.iterate to move through each data point.
        """
        # Checks to see if it needs to generate its own figures
        if self.plot and self.auto_figs and not self.figs_set:
            self.create_figs()
        
        # If plots should have been set but are not, return 0
        if self.plot is True and self.figs_set is False:
            return 0
        
        # Run the loop
        with self.meas.run() as datasaver:
            # Check if we are within the stopping condition
            while abs(self.setpoint - self.stop) > abs(self.step/2):
                self.iterate(datasaver)
        
        return 1
            
    def iterate(self, datasaver):
        """
        Runs one 'step' in the sweep. Takes in only the datasaver object, which is always
        a Measurement object's run() function (see autorun()). Iterate will update the 
        sweeping parameter, read each of the tracking parameters, and update the plots
        if plotting is enabled. Returns all values as a list of tuples, in the form of
        (parameter_name, parameter_value).
        """
        t = time.monotonic() - self.t0
        # Step the setpoint, and update the value
        self.setpoint = self.step + self.setpoint
        self.set_param.set(self.setpoint)
               
        # Update the sweeping parameter plot
        if self.plot is True:
            self.setaxline.set_xdata(np.append(self.setaxline.get_xdata(), t))
            self.setaxline.set_ydata(np.append(self.setaxline.get_ydata(), self.setpoint))
            self.setax.relim()
            self.setax.autoscale_view()
        
        # Pause if desired
        if self.inter_delay is not None:
            time.sleep(self.inter_delay)

        # Create our data storage object, which is a list of tuples of the parameter
        # and its value
        data = [
            (self.set_param, self.setpoint),
            ('time', t)
        ]
        
        # Loop through each of the tracking parameters
        for i, p in enumerate(self._params):
            # Update their values, and add them to the data object
            v = p.get()
            data.append((p, v))
            # Update each of the plots for the tracking parameters
            if self.plot is True:
                self.plines[i].set_xdata(np.append(self.plines[i].get_xdata(), self.setpoint))
                self.plines[i].set_ydata(np.append(self.plines[i].get_ydata(), v))
                self.axes[i].relim()
                self.axes[i].autoscale_view()
            
        # Add this point to the dataset
        datasaver.add_result(*data)
        
        # Set the plots
        if self.plot is True:        
            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.001)
            
        # Finally return all data
        return data
    
    def get_measurement(self):
        """
        Returns the measurement object.
        """
        return self.meas
    
    def save(self):
        """
        Saves the plots as a png
        (may not work? untested)
        """
        b = io.BytesIO()
        self.fig.savefig(b, format='png')






class SweepThread(QThread):
    """
    SweepThread uses QThread to separate data taking from the GUI, in order to still run.
    Written specifically for use in Sweep1DWindow.
    """
    # Two distinct signals created - one when thread is completed, one when the
    # plots ought to be updated in the GUI
    completed = pyqtSignal()
    update_plot = pyqtSignal()
    
    def __init__(self, parent, s):
        """
        Initializes the thread. Takes in the parent (which is generally Sweep1DWindow), and the
        Sweep1D object which is to be used.
        """
        self.parent = parent
        self.s = s
        self.data = []
        QThread.__init__(self)
        # Connect our signals to the functions in Sweep1DWindow
        self.completed.connect(self.parent.thread_finished)
        self.update_plot.connect(lambda: self.parent.update_plot(self.data))
        
    def __del__(self):
        """
        Standard destructor.
        """
        self.wait()
        
    def run(self):
        """
        Function to run the thread, and thus the sweep.
        """
        with self.parent.meas.run() as datasaver:
            # Always check to ensure that we want to continue running, aka we haven't paused
            while self.parent.running is True: 
                # Iterate the sweep
                self.data = self.s.iterate(datasaver)
                # Tell our parent what the data is, and tell it to update the plot
                self.parent.curr_val = self.data[0][1]
                self.update_plot.emit()
                # Check to see if our break condition has been met.
                if abs(self.parent.curr_val - self.parent.v_end) <= abs(self.parent.v_step/2):
                    self.completed.emit()
                    break



        































