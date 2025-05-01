#!/usr/env/bin python

import numpy
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats

class LigandEquib:

    def __init__(self):
        pass

    def info(self):
        text = """
        Sequential ligand binding model
               K1         K2
        P + L <-> PL + L <-> PL2

        Total protein (Ptotal) following mass conservation 
        Ptotal = P + PL + PL2 = 1

        Association constant
        K1 = PL/P*L

        Solve for P given K1 and K2
        P = 1 / (1+K1*L)
        """
        print(text)

    def fractions(self, L, Ks):
        '''
        L  = ligand concentration
        Ks = [K1, K2, ... ] array of K values

        Returns fractional abundance
        '''
        # setup
        L = numpy.array(L)
        
        # number of fractional species
        species = len(Ks) + 1           

        # calc numerator
        num = []
        den = 0.0
        for frac in range(species):
            if frac == 0:
                num.append(L*0. + 1.)
                den = L*0. + 1.
            else:
                Kt = 1.
                for k in Ks[:frac]:
                    Kt = Kt * k
                num.append(Kt * L**float(frac))
                den += Kt * L**float(frac)

        # now divide
        return num/den

    def free_lipid(self, L, Kagg):
        '''
        Determine the fraction of lipid available to bind given Ltotal.
        The Ltotal here is the back calculated Lfree but need to seperate the
        Lipid that can bind Lavail from those in aggregates. See 
        DOI 10.1021/jacs.6b01771 for more details.
        
        L  = ligand concentration
        Kagg = equilibrium assoc constant

        Returns fractional abundance
        '''
        num = -1. + numpy.sqrt(1.0 + 8.0*Kagg*numpy.array(L))
        den = 4.0*Kagg
        return num/den


class Fit(LigandEquib):

    def __init__(self, filename):
        
        # colour scheme
        self.my_colors = { 1 : "#ff9933", 2 : '#ffd700', \
                           3 : "#7fff00", 4 : '#00eeff', 5 : "#0331fc", \
                           6 : "#ba1e8c", 7 : '#b30505', 8 : "#fc4f05", \
                           9 : "#55966d" }
        self.def_color = "#808080"
        
        # get raw data
        raw = numpy.loadtxt(filename)

        # get Ptot
        self.Ptot = raw[:1:,0][0]
        raw = raw[1:]

        # extract [ligand]
        self.tmp_L  = raw[::,0]

        # extract and resort fraction concentrations
        raw     = raw[::,1:]
        self.eF = [ raw[::,index] for index in range(len(raw[0])) ]
        
        # calc Lfree
        self.L = []
        for Lo,row in zip(self.tmp_L,raw):
            Lbound = 0.0
            for n,PorPL in enumerate(row):
                Lbound += float(n)*PorPL*self.Ptot
            
            if Lo-Lbound < 0.0:
                print("ERROR Lfree negative", Lo-Lbound)
                self.L.append( 0.0 )
            else:
                self.L.append( Lo-Lbound )
            #print(Lo-Lbound)
        print(self.L)
                
####################################################################
    def fit_Ks(self, guess= 10 *10.**-6):
        # make a guess
####################################################################   
     
        p = [ 1./guess for i in range(len(self.eF)-1) ]
        #print(p)
        p[1] = 1./(1*10.**-9)
        p[1] = 1./(2*10.**-8)
        p[2] = 1./(4*10.**-7)
        p[3] = 1./(4*10.**-3)
        #p3 was **-3

        # now minimize scipy.fmin_X functions
        q = optimize.fmin_powell(self.err_func, p, disp=0)
        #q = optimize.fmin_bfgs(self.err_func, p, disp=0)
        
        # Alternatively use least squares
        #q = optimize.leastsq(self.err_func_array, p, args=(self.L))[0]

        # print stats
        print("Results from sequential binding model")
        for i,val in enumerate(1./numpy.array(q)):
            print("Kd%i (uM) = %.6f" % (i+1,val*10.0**6.0))
            
            # step = i+1
            # print(step)
            # micro = (val * ((4-step) + 1))/step
            # print("Kd%i (uM) = %.2f Kd_micro = %.2f" % (i+1,val*10.0**6.0, micro*10.0**6.0))
            
        chi = self.err_func(q)
        mod = self.fractions(self.L, q)
        r = stats.pearsonr( numpy.ndarray.flatten(numpy.array(mod)), numpy.ndarray.flatten(numpy.array(self.eF)) )[0]
        print("chi**2=%.2f, R**2=%.2f" % (chi, r**2))
        print()
        
        # generate hi-res data for theoretical fit
        hr_x = numpy.arange(0, 1.25*max(self.L), 0.1 * 10.**-6)
        hr_mod = self.fractions(hr_x, q)

        # now plot
        plt.figure(1)
        ind = 0
        X = numpy.array(self.L) * 10.**6
        for cf, ef in zip(hr_mod, self.eF):
            if ind == 0:
                color = self.def_color
            else:
                color = self.my_colors[ind%10]
            plt.plot(X, ef, "o", color=color)
            plt.plot(hr_x * 10**6, cf, "-", color=color)
            ind += 1
        plt.xlabel( '[ligand] (uM)' )
        plt.ylabel( 'mole fraction' )  
        plt.savefig("fit.svg", format='svg')
        plt.show()
        plt.close()

    def err_func(self, p, weight=False):
       calc = self.fractions(self.L, p)
       diff = numpy.absolute(calc-self.eF)
       
       if weight:
           # weighting function
           weight = numpy.ones(diff[0].shape)
           for i,j in enumerate(weight):
               if i <= 6: #int(len(weight)/2):
                           weight[i] = 2
           return numpy.sum(diff * weight)
       
       else:
           #return numpy.absolute(calc-self.eF).sum()
           return numpy.square(calc - self.eF).sum()

    def err_func_array(self, p, l):
       calc = self.fractions(self.L, p)
       err = self.L*0. + 1.
       for cf, ef in zip(calc, self.eF):
           err = err + (cf-ef)**2
       return err


    #--- including lipid aggregate model ---
    def complex_fit_Ks(self, guess= 5.*10.**-3, agg=1.*10.**-3):
        
        # make a guess for KPL's
        p = [ 1./guess for i in range(len(self.eF)-1) ]

        # append K for lipid aggregate model
        p.append( 1./agg )

        # now minimize scipy.fmin_X functions
        q = optimize.fmin_powell(self.err_func_agg, p, disp=0)
        #q = optimize.fmin_bfgs(self.err_func, p, disp=0)
        
        # Alternatively use least squares
        #q = optimize.leastsq(self.err_func_array, p, args=(self.L))[0]

        # setup
        q = list(q)

        # get chi
        chi = self.err_func_agg(q)
        
        # now reset L
        Kagg = q.pop()
        L = self.free_lipid(self.L, Kagg)
        #print('Kn', 1./Kagg)

        # calc model
        mod = self.fractions(L, q)

        # print stats
        print("Results from adduct model")
        for i,val in enumerate(1./numpy.array(q)):
            print("Kd%i (uM) = %.2f" % (i+1,val*10.0**6.0))
        print("Kagg (uM) = %.2f" % (10.**6/Kagg))
        r = stats.pearsonr( numpy.ndarray.flatten(numpy.array(mod)), numpy.ndarray.flatten(numpy.array(self.eF)) )[0]
        print("chi**2=%.2f, R**2=%.2f" % (chi, r**2))

        # generate hi-res data for theoretical fit
        hr_x = numpy.arange(0, max(L)*1.25, 0.1 * 10.**-6)
        hr_mod = self.fractions(hr_x, q)

        # now plot
        ind = 0
        plt.figure(2)
        X = numpy.array(L) * 10.**6
        for cf, ef in zip(hr_mod, self.eF):
            if ind == 0:
                color = self.def_color
            else:
                color = self.my_colors[ind%6]
            plt.plot(X, ef, "o", color=color)
            plt.plot(hr_x * 10**6, cf, "-", color=color)
            ind += 1
        plt.xlabel( '[Adduct] (uM)' )
        plt.ylabel( 'mole fraction' )
        plt.savefig("plot.svg", format='svg')
        plt.show()
        plt.close()
        
    def err_func_agg(self, p, n=2):
        p = list(p)
        Ka = p.pop()
        L = self.free_lipid(self.L, Ka)
        calc = self.fractions(L, p)
        return numpy.square(calc - self.eF).sum()
        

if __name__ == "__main__":
     
    # change filename below
    fname = r"c:\Users\kevan\Documents\Russell Group\Python Script\AAD2O_thermodynamics\temp_AA-D_15c_1.txt"
    #fname = r"D:\DATA\UHMR DATA\GroEL\GroEL_ATP_Thermo_ULTIMATE Compendium\BigDaddy Calc\5C-1.txt"
    fit = Fit(fname)

    # simple fit 
    fit.fit_Ks()
    
    # complex fit
    #fit.complex_fit_Ks()






    
