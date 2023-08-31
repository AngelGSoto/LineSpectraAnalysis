'''
Script to estimate the chi-square
Author: Luis A. Gutiérrez Soto
28/08/2023
'''
import numpy as np
from astropy.table import Table, QTable
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

def closest(lst, K):
    '''find the closest number'''
    lst = np.array(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]

def find_line(wl_vacuum, spec):
    '''
    This function find the line
    in a wavelenght range
    '''
    line_region = SpectralRegion((wl_vacuum-50)*u.AA, (wl_vacuum+50)*u.AA)
    line_spec = extract_region(spec, line_region)
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        line_spec = find_lines_derivative(line_spec, flux_threshold=0.0) # find the lines
        print("Number of line found:", len(line_spec))
    if len(line_spec) > 1:
        wl_line = closest(line_spec['line_center'].value, wl_vacuum)
        mask = line_spec['line_center'].value == wl_line
        line_spec_mask = line_spec[mask]
    else:
        line_spec_mask = line_spec
    return line_spec_mask

def measurent_line(wl_vacuum, spec, lamb, units_flux, text, NameLine, saveplot = "y"):
    '''
    Meassurent the flux line
    '''
    line_spec_mask = find_line(wl_vacuum,  spec)
    #Extract again the region using the lice center found
    line_region_ = SpectralRegion(line_spec_mask['line_center'] - 5.5 * u.AA, line_spec_mask['line_center'] + 5.5 * u.AA)
    sub_spectrum_line = extract_region(spec, line_region_)
    line_para_line = estimate_line_parameters(sub_spectrum_line, models.Gaussian1D())
    print("Parameters of the 1D-Gaussin:", line_para_line)
    # Fit the spectrum and calculate the fitted flux values (``y_fit``)
    g_init_line = models.Gaussian1D(amplitude=line_para_line.amplitude.value * units_flux,
                                    mean=line_para_line.mean.value * u.AA , stddev=line_para_line.stddev.value * u.AA )
    g_fit_line = fit_lines(spec, g_init_line, window=(line_spec_mask['line_center'] - 5 * u.AA, line_spec_mask['line_center'] + 5 * u.AA))
    y_fit_line = g_fit_line(lamb)
    #Integrating along the fit 1D-Gaussian
    gauss = Spectrum1D(spectral_axis=lamb, flux=y_fit_line) 
    sub_gauss = extract_region(gauss, line_region_)
    min_lamb = line_para_line.mean.value - 3*line_para_line.stddev.value
    max_lamb = line_para_line.mean.value + 3*line_para_line.stddev.value
    sub_region_int = SpectralRegion(min_lamb * u.AA,  max_lamb * u.AA)
    sub_gauss_int = extract_region(gauss, sub_region_int)
    flux_line = np.trapz(sub_gauss_int.flux, sub_gauss_int.spectral_axis) 
    #Ploting the lina and fit Gaussian
    if saveplot == "y":
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.plot(spec.spectral_axis, spec.flux, linewidth=10, c = "blueviolet", label = "Observed")
        plt.plot(spec.spectral_axis, y_fit_line, linewidth=10, c = "orange", linestyle='dashed', label = "1D Gaussian model")
        plt.xlabel('Wavelength $(\AA)$')
        #plt.ylim(-100, (sub_spectrum_line.max() + 500*units_flux))
        plt.xlim((line_spec_mask['line_center'].value-15), (line_spec_mask['line_center'].value+15))
        bbox_props = dict(boxstyle="round", fc="w", ec="0.88", alpha=0.6, pad=0.1)
        plt.text(0.1, 0.9, text,
             transform=ax.transAxes, c="black", weight='bold', fontsize=35, bbox=bbox_props)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(NameLine + ".pdf")
        plt.close()
    else:
        pass
    return flux_line


def ew(wl_vacuum, spec):
    '''
    Estimating the equivalent width of a line
    '''
    line_spec_mask = find_line(wl_vacuum,  spec)
    min_lamb = line_spec_mask['line_center'] - 5.5 * u.AA
    max_lamb = line_spec_mask['line_center'] + 5.5 * u.AA
    sub_region_int = SpectralRegion(min_lamb,  max_lamb)
    sub_spect_int = extract_region(spec, sub_region_int)
    #Estimating equivalent width
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        cont_norm_spec = spec / fit_generic_continuum(spec)(spec.spectral_axis)
    ew = equivalent_width(cont_norm_spec, regions=sub_region_int)
    return ew

def fwhm_luis(wl_vacuum, spec):
    '''
    Estimating fwhm of a line
    '''
    line_spec_mask = find_line(wl_vacuum,  spec)
    min_lamb = line_spec_mask['line_center'] - 5.5 * u.AA
    max_lamb = line_spec_mask['line_center'] + 5.5 * u.AA
    sub_region_int = SpectralRegion(min_lamb, max_lamb)
    sub_spect_int = extract_region(spec, sub_region_int)
    line_para_line = estimate_line_parameters(sub_spect_int, models.Gaussian1D())
    sub_spectrum_line = extract_region(spec, sub_region_int)
    fwhm_ = gaussian_fwhm(sub_spectrum_line)
    return fwhm_
        
def npx_errcont(wl_vacuum, spec):
    '''
    Find mean standar deviation both side of the line, and number of pixel cover for the 
    line
    '''
    line_spec_mask = find_line(wl_vacuum,  spec)
    min_lamb = line_spec_mask['line_center'] - 3.5 * u.AA
    max_lamb = line_spec_mask['line_center'] + 3.5 * u.AA
    sub_region_int = SpectralRegion(min_lamb, max_lamb)
    sub_spect_int = extract_region(spec, sub_region_int)
    line_para_line = estimate_line_parameters(sub_spect_int, models.Gaussian1D())
    sub_spectrum_line = extract_region(spec, sub_region_int)
    fwhm_ = gaussian_fwhm(sub_spectrum_line)
    min_lamb_ = line_para_line.mean.value - fwhm_.value / 2.
    max_lamb_ = line_para_line.mean.value + fwhm_.value / 2.
    sub_region_line_ = SpectralRegion(min_lamb_* u.AA,  max_lamb_* u.AA)
    sub_line_ = extract_region(spec, sub_region_line_)
    n_pixel = len(sub_line_.spectral_axis)
    print("Number of pixels:", n_pixel)
    # Determinante the median desviation standar in both side of the line
    min_lamb_cont = line_para_line.mean.value - fwhm_.value
    max_lamb_cont = line_para_line.mean.value + fwhm_.value
    min_lamb_cont_ = min_lamb_cont - 20
    max_lamb_cont_ = max_lamb_cont + 20
    sub_region_cont_left = SpectralRegion(min_lamb_cont_ * u.AA, min_lamb_cont * u.AA) # extract spec on left side of the line
    sub_region_cont_right = SpectralRegion(max_lamb_cont * u.AA, max_lamb_cont_ * u.AA) # extract spec on right side of the line
    sub_cont_left =  extract_region(spec, sub_region_cont_left)
    sub_cont_right =  extract_region(spec, sub_region_cont_right)
    err = []
    for errleft in sub_cont_left.uncertainty.array:
        err.append(errleft)
    for errright in sub_cont_right.uncertainty.array:
        err.append(errright)
    err = np.array(err).mean()
    # equivalent widht
    return n_pixel, err

def err_line(wl_vacuum, spec, D=1.0002302850208247):
    '''
    Estimate the uncertainty of the lines
    using the eq. of Tresse et al. 1999
    '''
    ew1 = ew(wl_vacuum, spec) #equivalent width
    npixel, err_cont = npx_errcont(wl_vacuum, spec) #number of pixel and error continuum
    err_line = err_cont * D * np.sqrt((2 * npixel) + (np.abs(ew1.value) / D))
    return err_line

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

#testing
# Defining units astropy
lamb = wl * u.AA 
flux = Flux_sum * u.Unit('erg cm-2 s-1 AA-1') 
spec = Spectrum1D(spectral_axis=lamb, flux=flux)


# Subtracting the continuum
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    g1_fit = fit_generic_continuum(spec)
y_continuum_fitted = g1_fit(spec.spectral_axis)
spec_sub = spec - y_continuum_fitted


# Emission lines
# lines = {"[NeIII]": 3868.760,
#          "[NeIII]H7": 3967.470,
#          "Hdelta": 4101.742,
#          "Hgamma": 4340.471,
#          "HeII": 4685.99,
#          "Hbeta": 4861.333,
#          "[OIII]4958": 4958.911,
#          "[OIII]5006": 5006.843,
#          "[FeIII]": 5412.12,
#          "Halpha": 6564.614,
#          "[ArV]": 7005.87,
#          "[ArIII]7135": 7135.80,
#          "[ArIII]7751": 7751.06,
#          "HeII8236": 8236.8
#          }

# lines = {"[NeIII]+H7": 3967.470,
#          "Hdelta": 4101.742,
#          "Hgamma": 4340.471,
#          "HeII4685": 4685.99,
#          "Hbeta": 4861.333,
#          "[OIII]4958": 4958.911,
#          "[OIII]5006": 5006.843,
#          "HeII5412": 5412.12,
#          "Halpha": 6564.614,
#          }

# lines_name = ["[Ne III]+H7",
#               r"H$\delta$",
#               r"H$\gamma$",
#               "He II$\lambda$4685", 
#               r"H$\beta$",
#               "[O III]$\lambda$4958",
#               "[O III]$\lambda$5006",
#               "He II$\lambda$5412", 
#               r"H$\alpha$"]

# lines = {"[OIII]5006": 5006.840,
#          "Halpha": 6562.770,
#          }

# lines_name = ["[O III]$\lambda$5006",
#               r"H$\alpha$"]

lines = {"5006": 5006.840,
         "5200": 5200.000,
         "5517": 5517.660,
         "5537": 5537.600,
         "5575": 5575.000,
         "5754": 5754.600,
         "5875": 5875.660,
         "6300": 6300.300,
         "6312": 6312.100,
         "6363": 6363.770,
         "6548": 6548.100,
         "6562": 6562.770,
         "6583": 6583.500,
         "6678": 6678.160,
         "6716": 6716.440,
         "6730": 6730.820,
         "7005": 7005.67,
         "7065": 7065.25,
         "7135": 7135.80,
         "7319": 7319.99,
         "7329": 7329.67,
         "7281": 7281.35,
         "7751": 7751.12,
         "8045": 8045.63,
         "8237": 8237.15,
         "8413": 8413.32,
         "8467": 8467.25,
         "8581": 8581.87,
         "8598": 8598.39,
         "8665": 8665.02,
         "8727": 8727.12,
         "8750": 8750.47,
         "8862": 8862.78}
                  

lines_name = ["5006",
              "5200",
"5517",
"5537",
"5575",
"5754",
"5875",
"6300",
"6312",
"6363",
"6548",
"6562",
"6583",
"6678",
"6716",
"6730",
"7005",
"7065",
"7135",
"7319",
"7329",
"7281",
"7751",
"8045",
"8237",
"8413",
"8467",
"8581",
"8598",
"8665",
"8727",
"8750",
         "8862"] 

nlines_ = []
lines_ = []
flux_lines = []
err_lines = []
EW = []
FWHM = []
for n, (v, t) in zip(lines_name, lines.items()):
    try:
        flux_lines.append(measurent_line(t, spec_sub, spec_sub.spectral_axis, u.Unit('erg cm-2 s-1 AA-1'), n, v, saveplot = "y").value)
    except ValueError:
        flux_lines.append(np.nan)
    EW.append(ew(t, spec))
    FWHM.append(fwhm_luis(t, spec))
    lines_.append(t)
    nlines_.append(v)


#creating table and save it
table = QTable([nlines_, lines_, flux_lines, EW, FWHM],
           names=('Line', 'Lambda', 'Flux', "EW", "FWHM"),
           meta={'name': 'first table'})
#save the table
file_name = file_.replace(".fits", "-lines.dat")
table.write(file_name, format="ascii", overwrite=True)

