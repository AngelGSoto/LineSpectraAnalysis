'''
Script to estimate the chi-square
Author: Luis A. Gutiérrez Soto
28/08/2023
'''
import numpy as np
from astropy.table import vstack, Table, QTable 
import matplotlib.pyplot as plt
import os
import pyCloudy as pc
from astropy.io import fits
import seaborn as sn
import argparse
import sys
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.wcs import WCS
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
# specutils packages
from specutils import Spectrum1D
from specutils.analysis import line_flux
from specutils.fitting import fit_generic_continuum
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from specutils.analysis import centroid
from specutils.analysis import moment
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.fitting import fit_lines
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi
from specutils.analysis import centroid
from specutils.fitting.continuum import fit_continuum
from specutils.fitting import fit_generic_continuum
import warnings
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
quantity_support()
sn.set_context("poster")


# read the files/spectra
parser = argparse.ArgumentParser(
    description="""Making the LAMOST spectra""")

parser.add_argument("source", type=str,
                    default="spec-56568-VB081N11V1_sp05-101",
                    help="Name of source-spectrum, taken the prefix")

parser.add_argument("--rowi", type=float,
                    default=406,
                    help="row initial to initianilized the sum")

parser.add_argument("--rowf", type=float,
                    default=443,
                    help="row final")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the fit")


cmd_args = parser.parse_args()
file_ = cmd_args.source + ".fits"
hdu = fits.open(file_)
hdudata = hdu[0].data
nx, wav0, i0, dwav = [hdu[0].header[k] for k in ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
wl = wav0 + (np.arange(nx) - (i0 - 1))*dwav

Flux = []
for i in range(int(cmd_args.rowi), int(cmd_args.rowf)):
    Flux.append(hdudata[i])
    #Flux = hdudata
Flux_arr = np.array(Flux)
# Calculate the sum of elements across all rows (sum along axis 0)
Flux_sum = np.sum(Flux_arr, axis=0)
Flux_sum_= Flux_sum/1e-14

#testing
# Defining units astropy
lamb = wl * u.AA 
flux = Flux_sum * u.Unit('erg cm-2 s-1 AA-1') 
spec = Spectrum1D(spectral_axis=lamb, flux=flux)
sub_region = SpectralRegion(3700*u.AA, 7300*u.AA)
sub_spec = extract_region(spec, sub_region)

# Subtracting the continuum
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    g1_fit = fit_generic_continuum(sub_spec)
y_continuum_fitted = g1_fit(sub_spec.spectral_axis)
spec_sub = sub_spec - y_continuum_fitted

#Spliting the spectrum en blue and red part
sub_region_blue = SpectralRegion(3700*u.AA, 5100*u.AA)
sub_region_red = SpectralRegion(5100*u.AA, 7300*u.AA)
sub_spectrum_blue = extract_region(spec_sub, sub_region_blue)
sub_spectrum_red = extract_region(spec_sub, sub_region_red)

#Find the lines
lines_blue = find_lines_derivative(sub_spectrum_blue, flux_threshold=-0.1e-16)
lines_red = find_lines_derivative(sub_spectrum_red, flux_threshold=-1.3e-16)


lines_total = vstack([lines_blue, lines_red])
#Mask emission
mask = lines_total["line_type"] == "emission"
lines_emiss = lines_total[mask]

f, ax = plt.subplots(figsize=(14,8))  
plt.plot(wl, Flux_sum_, linewidth=1)
for wll in lines_emiss["line_center"]:
    ax.axvline(wll, color='k', linewidth=0.4, alpha=0.7, linestyle='--')
plt.title("comb_PNc2_M1.fits")
plt.xlabel("Wavelength, Angstrom")
plt.ylabel("Flux, erg/cm2/s/A")
#plt.xlim(xmin=5140,xmax=5265)
plt.ylim(ymin=-0.1,ymax=2.5)
plt.savefig(file_.replace(".fits", ".pdf"))

############################################################################################################
# Measuring the lines ######################################################################################
############################################################################################################
lines_luis = []
Flux_luis = []
for line in lines_emiss["line_center"]:
    line_region_ = SpectralRegion(line - 5.5 * u.AA, line + 5.5 * u.AA)
    sub_spectrum_line = extract_region(spec_sub, line_region_)
    line_para_line = estimate_line_parameters(sub_spectrum_line, models.Gaussian1D())
    print("Parameters of the 1D-Gaussin:", line_para_line)
    # Fit the spectrum and calculate the fitted flux values (``y_fit``)
    g_init_line = models.Gaussian1D(amplitude=line_para_line.amplitude.value * u.Unit('erg cm-2 s-1 AA-1'),
                                    mean=line_para_line.mean.value * u.AA , stddev=line_para_line.stddev.value * u.AA )
    g_fit_line = fit_lines(spec_sub, g_init_line, window=(line - 5 * u.AA, line + 5 * u.AA))
    y_fit_line = g_fit_line(spec_sub.spectral_axis)
    #Integrating along the fit 1D-Gaussian
    gauss = Spectrum1D(spectral_axis=spec_sub.spectral_axis, flux=y_fit_line) 
    sub_gauss = extract_region(gauss, line_region_)
    if ~np.isnan(line_para_line.stddev.value): 
        min_lamb = line.value - 3*line_para_line.stddev.value
        max_lamb = line.value + 3*line_para_line.stddev.value
    else:
        min_lamb = line.value - 3*6.097436286971232
        max_lamb = line.value + 3*6.097436286971232
                 
    sub_region_int = SpectralRegion(min_lamb * u.AA,  max_lamb * u.AA)
    
    sub_gauss_int = extract_region(gauss, sub_region_int)
    flux_line = np.trapz(sub_gauss_int.flux, sub_gauss_int.spectral_axis)
    lines_luis.append(line.value)
    # Replace 0 with inf
    flux_line_inf = np.where(flux_line == 0.0, np.inf, flux_line)
    Flux_luis.append(flux_line_inf)


    #Ploting the lina and fit Gaussian
    if cmd_args.savefig:
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.plot(spec_sub.spectral_axis, spec_sub.flux, linewidth=10, c = "blueviolet", label = "Observed")
        plt.plot(spec_sub.spectral_axis, y_fit_line, linewidth=10, c = "orange", linestyle='dashed', label = "1D Gaussian model")
        plt.xlabel('Wavelength $(\AA)$')
        #plt.ylim(-100, (sub_spectrum_line.max() + 500*units_flux))
        plt.xlim((line.value-15), (line.value+15))
        bbox_props = dict(boxstyle="round", fc="w", ec="0.88", alpha=0.6, pad=0.1)
        plt.text(0.1, 0.9, str(int(line.value)),
             transform=ax.transAxes, c="black", weight='bold', fontsize=35, bbox=bbox_props)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(str(int(line.value)) + ".pdf")
        plt.close()


#creating table and save it
table = QTable([lines_luis, Flux_luis],
        names=('Lambda', 'Flux'),
        meta={'name': 'first table'})
#save the table
file_name = file_.replace(".fits", "-lines.dat")
table.write(file_name, format="ascii.commented_header",  overwrite=True)
    
    

















