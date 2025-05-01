#!/usr/env/bin python

import numpy, scipy
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from scipy.optimize import curve_fit
import sys
import statistics


class data:

    def __init__(self, txt):
        """
        Import data from path into the following format
        data[temp(K^-1)] = average of rows with error bars [(avg1, avg2, ..., avg7), (std1, std2, ..., std7)]
        """
        import os

        self.data = {}
        print("loading %s" % txt)

        raw = numpy.loadtxt(txt)
        # Sort raw data by the first column (Temperature)
        raw = raw[raw[:, 0].argsort()]

        temp_accumulate = None
        temp_count = 0
        last_temp = None
        temp_data_points = []  # List to store individual data points for each temperature

        # Go through sorted data and accumulate
        for dp in raw:
            temperature = float(dp[0])
            T = temperature + 273.15

            # Check if new temperature or first entry
            if T != last_temp and last_temp is not None:
                # Calculate average for the accumulated data
                avg_data = temp_accumulate / temp_count
                # Calculate standard deviation for the collected data points
                std_data = numpy.std(temp_data_points, axis=0)
                self.data[last_temp] = (avg_data, std_data)

                # Reset accumulation and data points list for the new temperature
                temp_accumulate = numpy.zeros_like(dp[1:])
                temp_data_points = []
                temp_count = 0

            # Initialize or accumulate data
            if temp_accumulate is None:
                temp_accumulate = numpy.zeros_like(dp[1:])

            temp_accumulate += dp[1:]
            temp_data_points.append(dp[1:])
            temp_count += 1
            last_temp = T

        # Handle the last set of data
        if temp_count > 0:
            avg_data = temp_accumulate / temp_count
            std_data = numpy.std(temp_data_points, axis=0)
            self.data[last_temp] = (avg_data, std_data)


        # the average & std for each temperature are recorded as two arrays in self.data
        print(self.data)

    def VHoff(self, x, dH, dS):
        '''
        linear van't Hoff

        dG = -RTln(K) = dH - TdS

        rearrange

        ln(K) = -dH/RT + dS/R
        '''
        # Gas constant
        R = 8.3144598*10**-3 # kJ*K**-1*mol**-1 

        # calculate and return
        return   ((-1.0*dH)/(R*x)) + (dS/R)

    

    def NLVHoff(self, x, dH, Cp): #To=None, LN_Ko=None, Ts=[]):
        '''
        Input assumed to be LN(KA)
        see doi: 10.1074/jbc.M808500200
        '''
        # Gas constant
        R = 8.3144598*10**-3 # kJ*K**-1*mol**-1

        # Setup reference values
        # typically at 298K but will use first temp in
        # dataset
        To    = self.ref[0]
        LN_Ko = self.ref[1]
        
        # Let X = T, Y = ln(Keq)
        A = (dH - (To*Cp)) / R
        B = (1./To) - (1./x)
        C = Cp/R

        # calc and return
        return (A*B) + (C*numpy.log(x/To)) + LN_Ko


    def plot_data(self, ax, x, y, yerr, h_x, h_ycalc, col):
        """
        Helper function to plot data on a given subplot axis
        """
        y_min = min(y)
        y_max = max(y)
        y_range = y_max - y_min
        padding = y_range
        y_min -= padding
        y_max += padding

        ax.plot(1000. / x, y, 'bo')
        ax.plot(1000. / h_x, h_ycalc, 'r--')
        ax.errorbar(1000. / x, y, yerr=yerr, fmt='bo', ecolor='cornflowerblue', capsize=3)
        
        ax.set_ylim(y_min, y_max)
        #ax.set_title(col)
        ax.set_ylabel("Ln(Ka)", fontsize=12, fontname="Arial", labelpad=10)
        ax.legend(("exp","fit"), fontsize=12, loc="upper left")
        ax.tick_params(axis='y', labelsize=12, width=2)
        ax.tick_params(axis='x', labelsize=12)
        
        #ax.set_title(col)


    def solve_nonlinear_stacked(self):
        """
        Use non-linear van't Hoff equation and create vertically stacked plots
        """
        # get the keys
        temps = sorted(self.data.keys())
        R = 8.3144598 * 10 ** -3  # kJ*K**-1*mol**-1

        print(temps)

        # Create vertically stacked subplots with shared x-axis
        num_cols = len(self.data[temps[0]][0])
        fig, axes = plt.subplots(num_cols, 1, figsize=(7, 5 * num_cols), sharex=True)

        for col in range(num_cols):
            print("=" * 20)
            print(col + 1)
            print("=" * 20)

            # x and y
            #y = [self.data[temp][col] for temp in temps]
            y = numpy.array([self.data[temp][0][col] for temp in temps])
            y_std_devs = numpy.array([self.data[temp][1][col] for temp in temps])

            x = [float(temp) for temp in temps]
            x = numpy.array(x)

            # set index here
            index = 3
            self.ref = x[index], y[index]

            # fit the data with error
            popt, pcov = curve_fit(self.NLVHoff, x, y)

            dH, cP = popt
            e_dH = numpy.sqrt(pcov[0, 0])
            e_cP = numpy.sqrt(pcov[1, 1])

            kd = 1. / numpy.exp(y)

            print('Kd = %.2f' % (kd[index] * 10. ** 6))
            print('To = %s' % (x[index] - 273.15))
            print('dH   = %.2f +/- %.2f kJ/mole' % (dH, e_dH))
            print('cP = %.2f +/- %.2f kJ/mole:K' % (cP, e_cP))

            dG = -R * x[index] * y[index]
            dS = (dH - dG) / x[index]

            print('dG(calc, ref) = %.2f kJ/mol' % (dG))
            print('-dTS(calc, ref) = %.3f kJ/mol' % (-1. * dS * x[index]))

            h_x = numpy.arange((min(x) - 1) * 10., (max(x) + 1) * 10.) / 10.
            ycalc = self.NLVHoff(x, dH, cP)
            h_ycalc = self.NLVHoff(h_x, dH, cP)

            print("R**2=%.2f" % stats.pearsonr(y, ycalc)[0] ** 2)
            print()
            print("=" * 20)
            print()

            # Use the subplot for the current column
            self.plot_data(axes[col], x, y, y_std_devs, h_x, h_ycalc, col + 1)

        for ax in axes:
            ax.set_title('')

        plt.subplots_adjust(hspace=0.01, wspace=0)

        plt.xlabel("1000/T(K^-1)", fontsize=12, fontname="Arial", labelpad=15)  # Set the x-axis label at the bottom
        # plt.tight_layout()
        plt.savefig("/home/cow/Chem-Code/Graphs/NLVH_SPE_Graphs/stacked_plots.svg", format='svg')
        plt.show()


if __name__=='__main__':

    #d = data("spr_data_global.txt")
    d = data("/home/cow/Chem-Code/File_Input/Lose_Files/LnKa_AA_d2o_2.txt")
    d.solve_nonlinear_stacked()
    sys.exit()
    


