#!/usr/env/bin python

import numpy, scipy
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from scipy.optimize import curve_fit
import sys
import os


class data:

    def __init__(self,txt):
        """
        Import data from path into the following format
        data[temp(K^-1)][Ka?] = [(avg,std), (avg,std),...]

        format:

        temp K lnKa_avg lnKa_stdev
        """

        # import all the txt files
        self.data = {}

        print("loading %s" % txt)

        # import txt file
        raw = numpy.loadtxt(txt)

        # go through and import the data
        for dp in raw:
            
            # 0. calc T(K)
            temperature =  float(dp[0])
            T = temperature + 273.15

            # check if in dict
            if T not in self.data:
                self.data[T] = {}

            # 1. Add data
            self.data[T] = dp[1:]

                 

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


    def solve_nonlinear(self):
        """
        Use non-linear van't Hoff equation

        ln(K) = ((dHo-To*Cp)/R)*(1/To-1/T) + Cp/R*LnT/To + ln(Ko)

        """
        # get the keys
        temps = sorted(self.data.keys())
        R = 8.3144598*10**-3 # kJ*K**-1*mol**-1

        for col in range(len(self.data[temps[0]])):
            print("="*20)
            print(col+1)
            print("="*20)

            # x and y
            y  = [self.data[temp][col] for temp in temps]
            x  = [float(temp) for temp in temps]
            x = numpy.array(x)

            # set index here
            index = 4
            self.ref = x[index], y[index]

            # fit the data with error
            #(see http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/
            # _static/15.%20Fitting%20models%20to%20data.html)
            popt, pcov = curve_fit(self.NLVHoff, x, y)#, sigma=ye)
            
            dH, cP = popt
            e_dH = numpy.sqrt(pcov[0,0])
            e_cP = numpy.sqrt(pcov[1,1])

            # generate Kd
            kd = 1./numpy.exp(y)

            #print numpy.sqrt(numpy.diag(pcov))
            print('Kd = %.2f' % (kd[index]*10.**6) )#, kde[index]*10.**6)
            print('To = %s' % (x[index]-273.15))
            print('dH   = %.2f +/- %.2f kJ/mole' % (dH,e_dH))
            print('cP = %.2f +/- %.2f kJ/mole:K' % (cP, e_cP))

            #back calculate dS
            dG = -R*x[index]*y[index]#numpy.log(x[index])
            dS = (dH-dG)/x[index]
            
            # propogate error here
            #e_dG = numpy.absolute( R*x[index]*(kde[index]/kd[index]) )
            #e_dS = numpy.sqrt(e_dH**2 + e_dG**2) / x[index]
            print('dG(calc, ref) = %.2f kJ/mol' % (dG))
            print('-dTS(calc, ref) = %.3f kJ/mol' % (-1.*dS*x[index]))

            # setup hi-res calc plot
            h_x = numpy.arange((min(x)-1)*10.,(max(x)+1)*10.)/10.
            ycalc = self.NLVHoff(x, dH, cP)

            # overwrite here to get the average fit
            #dH = -60.1
            #cP = -7.64
            h_ycalc = self.NLVHoff(h_x, dH, cP)
            print("R**2=%.2f" % stats.pearsonr(y, ycalc)[0]**2)
            print()
            print("="*20)
            print()

            # setting the y axis scale
            y_min = min(y)
            y_max = max(y)
            y_range = y_max - y_min

            # Add some padding to the y-axis limits
            padding = y_range/2
            y_min -= padding
            y_max += padding
            
            print(numpy.sum(numpy.sqrt( (ycalc-y)**2)))
            #plt.errorbar(1000./x, y, yerr=ye, fmt='o')
            plt.plot(1000./x, y, 'o')
            plt.plot(1000./h_x, h_ycalc, 'r--')
            plt.ylim(y_min, y_max)
            plt.title(col+1)
            plt.xlabel("1000/T(K^-1)")
            plt.ylabel("Ln(KA)")
            plt.legend(("fit", "exp"))
            plt.savefig("/home/cow/Chem-Code/Graphs/NLVH_FFP_Graphs/%s_24C.png" % (col+1))
            #plt.close()
            plt.show()



if __name__=='__main__':

    #d = data("spr_data_global.txt")
    d = data("File_Input/Lose_Files/LnKa_int.txt")
    d.solve_nonlinear()
#   d.solve_linear()
    sys.exit()