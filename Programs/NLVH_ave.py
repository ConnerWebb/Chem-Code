#!/usr/env/bin python

import numpy, scipy
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from scipy.optimize import curve_fit
import sys
import statistics



class data:

    def __init__(self,txt):
        """
        Import data from path into the following format
        data[temp(K^-1)][Ka?] = [(avg,std), (avg,std),...]

        format:

        temp K lnKa_avg lnKa_stdev
        """
        import os

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


    def solve_nonlinear(self):
        """
        Use non-linear van't Hoff equation

        ln(K) = ((dHo-To*Cp)/R)*(1/To-1/T) + Cp/R*LnT/To + ln(Ko)

        """
        # get the keys
        temps = sorted(self.data.keys())
        R = 8.3144598*10**-3 # kJ*K**-1*mol**-1

        print("=" * 20)
        print("calculate the mean and std")
        for temperature in temps:
            values = self.data[temperature][0:]
            self.data[temperature][0] = sum(values) / len(values)
            self.data[temperature][1] = statistics.pstdev(values)
            print('data at T=%.2fK is:  %.2f +/- %.2f ' % (temperature, self.data[temperature][0], self.data[temperature][1]))
        print("=" * 20)

        # x and y
        y = [self.data[temp][0] for temp in temps]
        x = [float(temp) for temp in temps]
        std = [self.data[temp][1] for temp in temps]
        x = numpy.array(x)

        # set index (reference temperature) here
        index = 3
        self.ref = x[index], y[index]

        # fit the data with error
        # (see http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/
        # _static/15.%20Fitting%20models%20to%20data.html)
        popt, pcov = curve_fit(self.NLVHoff, x, y)  # , sigma=ye)

        dH, cP = popt
        e_dH = numpy.sqrt(pcov[0, 0])
        e_cP = numpy.sqrt(pcov[1, 1])

        # generate Kd
        kd = 1. / numpy.exp(y)

        # print numpy.sqrt(numpy.diag(pcov))
        print("calculate the thermodynamics parameters")
        print('Kd = %.2f' % (kd[index] * 10. ** 6))  # , kde[index]*10.**6)
        print('To = %s' % (x[index] - 273.15))
        print('dH   = %.2f +/- %.2f kJ/mole' % (dH, e_dH))
        print('cP = %.2f +/- %.2f kJ/mole:K' % (cP, e_cP))

        # back calculate dS
        dG = -R * x[index] * y[index]  # numpy.log(x[index])
        dS = (dH - dG) / x[index]

        # propogate error here
        # e_dG = numpy.absolute( R*x[index]*(kde[index]/kd[index]) )
        # e_dS = numpy.sqrt(e_dH**2 + e_dG**2) / x[index]
        print('dG(calc, ref) = %.2f kJ/mol' % (dG))
        print('-TdS(calc, ref) = %.3f kJ/mol' % (-1. * dS * x[index]))

        # setup hi-res calc plot
        h_x = numpy.arange((min(x) - 1) * 10., (max(x) + 1) * 10.) / 10.
        ycalc = self.NLVHoff(x, dH, cP)

        # overwrite here to get the average fit
        # dH = -60.1
        # cP = -7.64
        h_ycalc = self.NLVHoff(h_x, dH, cP)
        print("R**2=%.2f" % stats.pearsonr(y, ycalc)[0] ** 2)
        print()
        print("=" * 20)
        print()

        # ye = [0.089,0.082,0.026,0.07,0.04]
        # setting the y axis scale
        y_min = min(y)
        y_max = max(y)
        y_range = y_max - y_min

        # Add some padding to the y-axis limits
        padding = (2-y_range)/2
        y_min -= padding
        y_max += padding

       
        
        # print("std is: %.2f" % numpy.sum(numpy.sqrt((ycalc - y) ** 2)))
        plt.errorbar(1000./x, y, yerr=std, fmt='o')
        # plt.plot(1000. / x, y, 'o')
        plt.plot(1000. / h_x, h_ycalc, 'r--')
        plt.ylim(y_min, y_max)
        #plt.title(col)
        plt.xlabel("1000/T(K^-1)")
        plt.ylabel("Ln(KA)")
        plt.legend(("fit", "exp"))
        plt.savefig("combined_25C.png")
        # plt.close()
        plt.show()

            
        
    def solve_linear(self):
                
        # now start solving

        # get the keys
        temps = sorted(self.data.keys())   

        for col in range(len(self.data[temps[0]])):
            print(col)

            # x and y
            y  = [self.data[temp][col] for temp in temps]
            x  = [float(temp) for temp in temps]
            
            # set temp to 1/T
            x = 1./numpy.array(x)

            # fit the data with error
            #(see http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/
            # _static/15.%20Fitting%20models%20to%20data.html)
            popt, pcov = curve_fit(self.VHoff, x, y)#, sigma=ye)

            dH, dS = popt
            e_dH = numpy.sqrt(pcov[0,0])
            e_dS = numpy.sqrt(pcov[1,1])

            #print numpy.sqrt(numpy.diag(pcov))
            print('dH   = %.2f +/- %.2f kJ/mole' % (dH,e_dH))
            print('-TdS = %.4f +/- %.4f kJ/mole' % (-298.*dS, -298.*e_dS))
            print("dG   = %.2f kJ/mole" % (dH + (-298.*dS)))
            print()

            ycalc = self.VHoff(x, dH, dS)
            print(numpy.sum(numpy.sqrt( (ycalc-y)**2)))
            print("R**2=%.2f" % stats.pearsonr(y, ycalc)[0]**2)
            plt.plot(x, y, 'o')
            plt.plot(x, ycalc)
            
            plt.show()            
            
            
        
            

if __name__=='__main__':

    #d = data("spr_data_global.txt")
    # d = data("temp.txt")
    d = data("/home/cow/Chem-Code/Lose_Files/temp.txt")
    d.solve_nonlinear()
#   d.solve_linear()
    sys.exit()
    
    # # some data here
    # LN_KA = numpy.array( [13.63,13.31,13.20,13.01,12.80,12.59], dtype=float )
    # TEMP  = numpy.array( [16, 19, 22, 25, 28, 31], dtype=float )
    #
    # # make some fake error
    # e = numpy.repeat(0.5, len(LN_KA))
    #
    # # convert temp to Kelvin
    # TEMP = TEMP + 273.15
    # x = 1./TEMP
    #
    # #d.ref = x[0], LN_KA[0]
    #
    # # fit the data with error
    # #(see http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/
    # # _static/15.%20Fitting%20models%20to%20data.html)
    # popt, pcov = curve_fit(d.NLVHoff, x, LN_KA, sigma=e)
    #
    # dH, dS = popt
    # e_dH = numpy.sqrt(pcov[0,0])
    # e_dS = numpy.sqrt(pcov[1,1])
    #
    # #print numpy.sqrt(numpy.diag(pcov))
    # print('%f +/- %f' % (dH, e_dH))
    # print('%f +/- %f' % (dS, e_dS))
    #
    # plt.errorbar(x, LN_KA, yerr=e)
    # plt.plot(x, d.NLVHoff(x, dH, dS))
    # plt.show()
    

