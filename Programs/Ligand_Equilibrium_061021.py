#!/usr/env/bin python

import os
import csv
import numpy
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import optimize
from scipy import stats


print("\n")


# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', default='/home/cow/Chem-Code/File_Input/AAD2O_thermodynamics')
parser.add_argument('--csv-out-dir', default='/home/cow/Chem-Code/File_Output/Ligand_Equilibrium_061021')
parser.add_argument('--svg-out-dir', default='/home/cow/Chem-Code/Graphs/AAD20_Graphs')
parser.add_argument('--model-type', choices=['simple', 'complex'], default='simple')
parser.add_argument('--initial-guess-range', nargs=2, type=float, metavar=('MIN_COEFF', 'MAX_COEFF'), default=[1.0, 10.0],
                    help='Range of coefficients for initial guesses (e.g., --initial-guess-range 1 9)')
parser.add_argument('--initial-guess-exponent', type=int, default=-6,
                    help='Exponent to apply to all guesses (e.g., -6 for microM)')
parser.add_argument('--guess-steps', type=int, default=1000,
                    help='Number of steps to try in the initial guess range')
parser.add_argument('--no-show', action='store_true')
args = parser.parse_args()
# Add fallback for guess_steps
guess_steps = getattr(args, 'guess_steps', 1000)


def select_input_file(directory):

    print(f"Looking for input files in: {directory}\n")
    
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    
    if not files:
        print("No files found in the directory.")
        exit(1)
    
    for i, file in enumerate(files):
        print(f"[{i+1}] {file}")
    
    while True:
        try:
            choice = int(input("\nEnter the number of the file you want to use: ")) - 1
            if 0 <= choice < len(files):
                selected = os.path.join(directory, files[choice])
                print(f"\nSelected file: {selected}\n")
                return selected
            else:
                print("Invalid selection. Please choose a valid number.")
        except ValueError:
            print("Please enter a valid number.")

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
        return numpy.array(num)/den

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
        self.filename = filename
        self.base_name = os.path.splitext(os.path.basename(filename))[0]
        self.base_name = self.base_name.split('_', 1)[-1]  # remove prefix before first '_'
        self.date_stamp = datetime.today().strftime('%Y-%m-%d')

        self.my_colors = { 1 : "#ff9933", 2 : '#ffd700', 3 : "#7fff00", 4 : '#00eeff', 5 : "#0331fc", 6 : "#ba1e8c", 7 : '#b30505', 8 : "#fc4f05", 9 : "#55966d" }
        self.def_color = "#808080"

        raw = numpy.loadtxt(filename)
        self.Ptot = raw[:1:,0][0]
        raw = raw[1:]
        self.tmp_L  = raw[::,0]
        raw     = raw[::,1:]
        self.eF = [ raw[::,index] for index in range(len(raw[0])) ]

        self.L = []
        for Lo,row in zip(self.tmp_L,raw):
            Lbound = 0.0
            for n,PorPL in enumerate(row):
                Lbound += float(n)*PorPL*self.Ptot
            self.L.append(max(0.0, Lo-Lbound))

    def fit_Ks(self, guess_range):
        min_coeff, max_coeff = guess_range
        fixed_exp = args.initial_guess_exponent

        # Adjusting step size to ensure no skipping
        guess_steps = args.guess_steps  # You can also set a specific step size manually
        step_size = (max_coeff - min_coeff) / guess_steps
        guesses = numpy.arange(min_coeff, max_coeff + step_size, step_size) * (10 ** fixed_exp)

        best_r2 = -1
        best_params = None
        best_kd_data = []
        best_chi = None
        best_mod = None
        best_guess = None  # Variable to store the best guess

        for guess in guesses:
            p = [1. / guess for _ in range(len(self.eF) - 1)]
            custom_values = {
                1: 1. / (2e-8),
                2: 1. / (4e-7),
                3: 1. / (4e-3)
            }

            for idx, val in custom_values.items():
                if idx < len(p):
                    p[idx] = val

            q = optimize.fmin_powell(self.err_func, p, disp=0)
            mod = self.fractions(self.L, q)
            r = stats.pearsonr(numpy.ndarray.flatten(numpy.array(mod)),
                               numpy.ndarray.flatten(numpy.array(self.eF)))[0]
            r2 = r ** 2
            print(f"Tested guess: {guess:.2e}, R²: {r2:.4f}")

            # Check if this guess gives a better R² value
            if r2 > best_r2:
                best_r2 = r2
                best_guess = guess  # Store the best guess
                best_params = q
                best_chi = self.err_func(q)
                best_mod = mod
                best_kd_data = [[i + 1, val * 1e6] for i, val in enumerate(1. / numpy.array(q))]

        # After the loop, print out the best guess and its R²
        print(f"\nBest tested guess: {best_guess:.2e}, with R²: {best_r2:.4f}")

        # Output results for best fit
        print("Best fit:")
        for i, kd in best_kd_data:
            print(f"Kd{i} (uM) = {kd:.6f}")
        print(f"chi**2 = {best_chi:.2f}, R**2 = {best_r2:.2f}")



        # Save CSV and Plot (unchanged)
        output_dir_csv = args.csv_out_dir
        output_dir_svg = args.svg_out_dir
        os.makedirs(output_dir_csv, exist_ok=True)
        os.makedirs(output_dir_svg, exist_ok=True)

        csv_path = os.path.join(output_dir_csv, f"{self.base_name}_kd_values_{self.date_stamp}.csv")
        svg_path = os.path.join(output_dir_svg, f"{self.base_name}_fit_{self.date_stamp}.svg")

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Kd", "uM"])
            writer.writerows(best_kd_data)

        hr_x = numpy.arange(0, 1.25 * max(self.L), 0.1 * 1e-6)
        hr_mod = self.fractions(hr_x, best_params)

        plt.figure(1)
        ind = 0
        X = numpy.array(self.L) * 1e6
        for cf, ef in zip(hr_mod, self.eF):
            color = self.def_color if ind == 0 else self.my_colors[ind % 10]
            plt.plot(X, ef, "o", color=color)
            plt.plot(hr_x * 1e6, cf, "-", color=color)
            ind += 1
        plt.xlabel('[ligand] (uM)')
        plt.ylabel('mole fraction')
        plt.savefig(svg_path, format='svg')
        if not args.no_show:
            plt.show()
        plt.close()



    def err_func(self, p, weight=False):
        calc = self.fractions(self.L, p)
        diff = numpy.absolute(calc-self.eF)
        if weight:
            weight = numpy.ones(diff[0].shape)
            for i in range(len(weight)):
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
    def complex_fit_Ks(self, guess=5.*10.**-3, agg=1.*10.**-3):
        # make a guess for KPL's
        p = [ 1./guess for i in range(len(self.eF)-1) ]
        # append K for lipid aggregate model
        p.append(1./agg)
        # now minimize scipy.fmin_X functions

        q = optimize.fmin_powell(self.err_func_agg, p, disp=0)
        #q = optimize.fmin_bfgs(self.err_func, p, disp=0)
        
        # Alternatively use least squares
        #q = optimize.leastsq(self.err_func_array, p, args=(self.L))[0]
        q = list(q)
        chi = self.err_func_agg(q)
        Kagg = q.pop()
        L = self.free_lipid(self.L, Kagg)
        mod = self.fractions(L, q)

        print("Results from adduct model")
        for i,val in enumerate(1./numpy.array(q)):
            print("Kd%i (uM) = %.2f" % (i+1,val*10.0**6.0))
        print("Kagg (uM) = %.2f" % (10.**6/Kagg))
        r = stats.pearsonr(numpy.ndarray.flatten(numpy.array(mod)), numpy.ndarray.flatten(numpy.array(self.eF)))[0]
        print("chi**2=%.2f, R**2=%.2f" % (chi, r**2))

        hr_x = numpy.arange(0, max(L)*1.25, 0.1 * 10.**-6)
        hr_mod = self.fractions(hr_x, q)

        plt.figure(2)
        ind = 0
        X = numpy.array(L) * 10.**6
        for cf, ef in zip(hr_mod, self.eF):
            color = self.def_color if ind == 0 else self.my_colors[ind%6]
            plt.plot(X, ef, "o", color=color)
            plt.plot(hr_x * 10**6, cf, "-", color=color)
            ind += 1
        plt.xlabel('[Adduct] (uM)')
        plt.ylabel('mole fraction')
        svg_path = os.path.join(args.svg_out_dir, f"{self.base_name}_fit_{self.date_stamp}.svg")
        plt.savefig(svg_path, format='svg')
        if not args.no_show:
            plt.show()
        plt.close()

    def err_func_agg(self, p, n=2):
        p = list(p)
        Ka = p.pop()
        L = self.free_lipid(self.L, Ka)
        calc = self.fractions(L, p)
        return numpy.square(calc - self.eF).sum()

if __name__ == "__main__":
    fname = select_input_file(args.input_dir)
    fit = Fit(fname)

    if args.model_type == 'simple':
        fit.fit_Ks(args.initial_guess_range)
    elif args.model_type == 'complex':
        fit.complex_fit_Ks()
