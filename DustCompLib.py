#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
#%%
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.ticker as plticker
from scipy.optimize import nnls
# from dust_extinction.averages import G21_MWAvg
# ext = G21_MWAvg()
# import astropy.units as u
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# plot style configuration
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams.update({'font.size': 13})

################################################################
# function definitions
################################################################

h = 6.62606957e-34; #(J s)
kboltz = 1.3806488e-23; #(J K-1)
c0 = 299792458.0; #(m s-1)
au = 1.495978707e11 # (m)
rsun = 6.9634e8 # (m)
parsec = 3.0857e16 # (m)
sigma_SB = 5.67037e-8 #(W m−2 K−4)
C1 = 2*h/c0**2
C2 = h/kboltz

@njit(fastmath=True)
def B_nu_array(T, nu):
    nnu = len(nu)
    nT = len(T)
    out = np.empty((nnu, nT))

    for i in range(nnu):
        for j in range(nT):
            x = C2 * nu[i] / T[j]
            out[i, j] = C1 * nu[i]**3 / (np.exp(x) - 1.0)

    return out

@njit(fastmath=True)
def B_nu_scalar(T, nu):
    nnu = len(nu)
    out = np.empty(nnu)

    for i in range(nnu):
        x = C2 * nu[i] / T
        out[i] = C1 * nu[i]**3 / (np.exp(x) - 1.0)

    return out

@njit(fastmath=True)
def hlr_numba(radius, I_nu_map):
    nnu, nr = I_nu_map.shape
    hlr = np.zeros(nnu)

    for i in range(nnu):
        # compute cumulative flux
        cum = np.zeros(nr)
        for j in range(1, nr):
            dr = radius[j] - radius[j-1]

            f1 = 2*np.pi*radius[j-1]*I_nu_map[i, j-1]
            f2 = 2*np.pi*radius[j]*I_nu_map[i, j]

            cum[j] = cum[j-1] + 0.5 * (f1 + f2) * dr

        total = cum[-1]
        half = 0.5 * total

        # find radius
        for j in range(1, nr):
            if cum[j] >= half:
                f1 = cum[j-1]
                f2 = cum[j]
                r1 = radius[j-1]
                r2 = radius[j]

                t = (half - f1) / (f2 - f1)
                hlr[i] = r1 + t * (r2 - r1)
                break

    return hlr

def temp_at_hlr(radius, T_profile, hlr):
    T_hlr = np.zeros_like(hlr)
    for i in range(len(hlr)):
        T_hlr[i] = np.interp(hlr[i], radius, T_profile)
    return T_hlr

import numpy as np

def hlr_combined(
    radius_dust, I_dust,
    radius_rim, I_rim,
    radius_mid, I_mid
):
    """
    Compute half-light radius combining components with different radial grids.

    Parameters
    ----------
    radius_* : 1D arrays
        Radial grids for each component
    I_* : 2D arrays (n_freq, n_radius)
        Intensity maps

    Returns
    -------
    hlr : array (n_freq,)
        Half-light radius per frequency
    """

    def cumulative_flux(radius, I):
        integrand = 2 * np.pi * radius[None, :] * I

        cum = np.cumsum(
            0.5 * (integrand[:, 1:] + integrand[:, :-1]) *
            (radius[1:] - radius[:-1])[None, :],
            axis=1
        )

        # prepend zero
        cum = np.hstack([np.zeros((cum.shape[0], 1)), cum])
        return cum

    # cumulative flux per component
    cum_dust = cumulative_flux(radius_dust, I_dust)
    cum_rim  = cumulative_flux(radius_rim,  I_rim)
    cum_mid  = cumulative_flux(radius_mid,  I_mid)

    n_freq = I_dust.shape[0]
    hlr = np.zeros(n_freq)

    # common radius grid (union)
    radius_all = np.sort(np.unique(
        np.concatenate([radius_dust, radius_rim, radius_mid])
    ))

    for i in range(n_freq):

        # interpolate cumulative flux onto common grid
        F_d = np.interp(radius_all, radius_dust, cum_dust[i], left=0, right=cum_dust[i,-1])
        F_r = np.interp(radius_all, radius_rim,  cum_rim[i],  left=0, right=cum_rim[i,-1])
        F_m = np.interp(radius_all, radius_mid,  cum_mid[i],  left=0, right=cum_mid[i,-1])

        F_tot = F_d + F_r + F_m

        F_half = 0.5 * F_tot[-1]

        # find crossing
        idx = np.searchsorted(F_tot, F_half)

        if idx == 0:
            hlr[i] = radius_all[0]
        else:
            r1, r2 = radius_all[idx-1], radius_all[idx]
            f1, f2 = F_tot[idx-1], F_tot[idx]

            t = (F_half - f1) / (f2 - f1)
            hlr[i] = r1 + t * (r2 - r1)

    return hlr

@njit(fastmath=True)
def integrate_disk(I, radius):
    nnu, nr = I.shape
    out = np.zeros(nnu)

    for i in range(nnu):
        s = 0.0
        for j in range(nr - 1):
            r1 = radius[j]
            r2 = radius[j+1]

            dr = r2 - r1

            f1 = 2.0*np.pi*r1*I[i,j]
            f2 = 2.0*np.pi*r2*I[i,j+1]

            s += 0.5 * (f1 + f2) * dr

        out[i] = s
    return out

@njit(fastmath=True)
def make_radius_grid(r_in, r_out, n_r,t):

    #t = np.linspace(0.0, 1.0, n_r)
    radius = np.empty(n_r)

    ratio = r_out / r_in
    for i in range(n_r):
        radius[i] = r_in * ratio**t[i]

    return radius

@njit(fastmath=True)
def temperature_profile(radius, T0, q):
    n = len(radius)
    out = np.empty(n)

    inv_r0 = 1.0 / radius[0]

    for i in range(n):
        out[i] = T0 * (radius[i]*inv_r0)**q

    return out

@njit(fastmath=True)
def model_numba(theta, freq, kappa_arr,flux_star,n_r,n_rim,x,x_rim):
    # ---- unpack parameters ----
    r_in      = theta[0]
    r_out     = theta[1]

    T_dust_in = theta[2]
    q_dust    = theta[3]

    T_mid_in  = theta[4]
    q_mid     = theta[5]

    T_rim_in  = theta[6]
    q_rim     = theta[7]

    scale     = theta[8] #1e26 / (dscale*dscale)
    rim_width = theta[9]

    # r_star    = theta[10]
    # T_star    = theta[11]

    # ---- grid ----
    radius = make_radius_grid(r_in, r_out,n_r, x)
    radius_rim = make_radius_grid(r_in, r_in+rim_width,n_rim, x_rim)

    # ---- temperatures ----
    T_dust = temperature_profile(radius, T_dust_in, q_dust)
    T_mid  = temperature_profile(radius, T_mid_in, q_mid)
    T_rim  = temperature_profile(radius_rim, T_rim_in, q_rim)

    # ---- Planck ----
    I_mid = B_nu_array(T_mid, freq)
    I_rim = B_nu_array(T_rim, freq)
    I_dust = B_nu_array(T_dust, freq)

    # ---- integrate ----
    flux_mid = scale*integrate_disk(I_mid, radius)
    flux_rim = scale*integrate_disk(I_rim, radius_rim)

    # ---- star ----
    # Bstar = B_nu_scalar(T_star, freq)

    # flux_star = np.empty(len(freq))
    # for i in range(len(freq)):
    #     flux_star[i] = scale * (r_star*rsun/au)**2 * np.pi * Bstar[i]

    source_dust = integrate_disk(I_dust, radius)

    # ---- build NNLS matrix ----
    nnu, nk = kappa_arr.shape
    dust_arr = np.empty((nnu, nk))

    for i in range(nnu):
        for j in range(nk):
            dust_arr[i, j] = kappa_arr[i, j] * scale * source_dust[i]

    return flux_star,flux_rim,flux_mid,dust_arr

def get_component_fluxes(freq,param_dic,kappa,fit_mode,fluxdata):

    kappa_arr,kappa_label_list = kappa
    theta = np.array([\
        param_dic['r_in']['value'],
        param_dic['r_out']['value'],
        param_dic['T_dust_in']['value'],
        param_dic['q_dust']['value'],
        param_dic['T_midplane_in']['value'],
        param_dic['q_midplane']['value'],
        param_dic['T_rim_in']['value'],
        param_dic['q_rim']['value'],
        param_dic['scale']['value'],
        param_dic['rim_width']['value'],
        param_dic['r_star']['value'],
        param_dic['T_star']['value'],
    ])
    n_r = param_dic['n_r']['value']
    n_rim = param_dic['n_rim']['value']
    flux_star = param_dic['flux_star']['value']
    x = param_dic['x']['value']
    x_rim = param_dic['x_rim']['value']

    flux_star,flux_rim,flux_midplane,dust_arr = model_numba(theta, freq, kappa_arr,flux_star,n_r,n_rim,x,x_rim) #dust_arr

    try:
        dust_coeffs = nnls(dust_arr ,fluxdata-flux_star-flux_rim-flux_midplane)[0]
        #print(dust_coeffs)
    except RuntimeError:
        dust_coeffs = np.ones((len(kappa_label_list)))   
    except ValueError:
        dust_coeffs = np.ones((len(kappa_label_list))) 
    #print(kappa_label_list)
    for i,kappa_label in enumerate(kappa_label_list):
        param_dic[kappa_label]['value'] = dust_coeffs[i]
        #print(param_dic[kappa_label]['value'])
    flux_dust = dust_arr*dust_coeffs[None,:] #np.nansum( ,axis=1)
    # flux_dust = kappa_arr*dust_coeffs[None,:]*source_dust[:,None]
    #print('inside:',flux_dust)

    return flux_dust,flux_midplane,flux_star,flux_rim

def model_fn(freq,param_dic,kappa,fit_mode,fluxdata):
    #experimental: de-extinct the data
    #flux_new = fluxdata/ext.extinguish(wl*10000*u.AA, Av=param_dic['AV']['value'])
    #fluxdata = flux_new
    flux_dust,flux_midplane,flux_star,flux_rim = get_component_fluxes(freq,param_dic,kappa,fit_mode,fluxdata)
    # print(flux_dust+flux_midplane+flux_star+flux_rim)
    return np.nansum(flux_dust,axis=1)+flux_midplane+flux_star+flux_rim

# parametrisation (for dynesty)
def prior_transform(uniform_samples,priors): #param_dic,free_param_labels):
    """The transformation for uniform sampled values to the
    uniform parameter space."""
    #priors = np.array([(param_dic[key]['limits'][0], param_dic[key]['limits'][1]) for key in free_param_labels])
    return priors[:, 0] + (priors[:, 1] - priors[:, 0])*uniform_samples

# for dynesty
def lnlike(free_param_values, freq, fluxdata, fluxerr,other_args):
    param_dic,free_param_labels,kappa,fit_mode = other_args
    for (pvalue,label) in zip(free_param_values,free_param_labels):
        param_dic[label]['value'] = pvalue
    model = model_fn(freq,param_dic,kappa,fit_mode,fluxdata)
    #experimental: de-extinct the data
    #flux_new = fluxdata/ext.extinguish(wl*10000*u.AA, Av=param_dic['AV']['value'])
    #fluxdata = flux_new
    return -0.5 * np.nansum((fluxdata - model)**2 / fluxerr**2) / fluxdata.size

# # for emcee
# def lnprob(free_param_values, freq, fluxdata, fluxerr,other_args):
#     param_dic,free_param_labels,kappa = other_args
#     for (pvalue,label) in zip(free_param_values,free_param_labels):
#         low, up =  param_dic[label]['limits']
#         #print(label,low ,pvalue , up)
#         if not low < pvalue < up:
#             return -np.inf
#         param_dic[label]['value'] = pvalue
#         #print(label,pvalue ,param_dic[label]['value'])

#     model = model_fn(freq,param_dic,kappa)
#     # print(fluxdata-model)
#     # print(fluxdata)
#     # print(model)
#     # print((np.sum((fluxdata-model)**2/(fluxerr**2) )/len(fluxdata)))
#     return -0.5*(np.nansum((fluxdata-model)**2/(fluxerr**2) )/len(fluxdata))

def get_linestyle(kappa_label):
    if 'rv0.1' in kappa_label or '0.10um' in kappa_label or '0.1.' in kappa_label:
        linestyle = '--'
    elif 'rv1.0' in kappa_label or '1.00um' in kappa_label or '1.0.' in kappa_label:
        linestyle = (0, (5, 10)) #'loosely dashed'   
    elif 'rv2.0' in kappa_label or '2.00um' in kappa_label or '2.0.' in kappa_label:
        linestyle = (5, (10, 3)) #'long dash with offset'
    elif 'rv3.0' in kappa_label or '3.00um' in kappa_label or '3.0.' in kappa_label:
        linestyle = '-.'
    elif 'rv4.0' in kappa_label or '4.00um' in kappa_label or '4.0.' in kappa_label:
        linestyle = (0, (3, 5, 1, 5, 1, 5))  #'dashdotdotted'
    elif 'rv5.0' in kappa_label or '5.00um' in kappa_label or '5.0.' in kappa_label:
        linestyle = ':'
    else:
        linestyle = (0, (5, 10)) #'loosely dashed'       
    return linestyle

def get_color_label(kappa_label):
    color_table = [\
        ['Am_Mgolivine_Jae',r'Am. $\mathrm{Mg}_{2}\mathrm{SiO}_{4}$','blue'],
        ['Am_Mgol_Jae',r'Am. $\mathrm{Mg}_{2}\mathrm{SiO}_{4}$','blue'],
        ['MgOlivine',r'Am. $\mathrm{Mg}_{2}\mathrm{SiO}_{4}$','blue'],
        ['Am_MgFeolivine_Dor','Am. MgFe-olivine','royalblue'],
        ['Am_Ol-Mg50','Am. MgFe-olivine','royalblue'],
        ['Am_Ol-Mg40','Am. MgFe-olivine','royalblue'],
        ['Fay_Fabian','Fayalite','deepskyblue'],
        ['Am_Mgpyroxene_Dor',r'Am. MgSiO$_3$','darkorange'],
        ['Am_Mgpyr_Dor',r'Am. MgSiO$_3$','darkorange'],
        ['MgPyroxene',r'Am. MgSiO$_3$','darkorange'],
        ['Am_MgFepyroxene_Dor','Am. MgFe-pyroxene','gold'],
        ['MgFepyroxene_Dor','Am. MgFe-pyroxene','gold'],
        ['MgFepyr','Am. MgFe-pyroxene','gold'],
        ['Fo_Zeidler', 'Forsterite','lime'],
        ['Fo_Suto', 'Forsterite','lime'],
        ['Fo_Sogawa', 'Forsterite', 'lime'],
        ['Fo_Servoin', 'Forsterite', 'lime'],
        ['Koike_Fo100', 'Forsterite', 'lime'],
        ['Fors_aerosol', 'Forsterite', 'lime'],
        ['En_Zeidler','Enstatite','orangered'],
        ['Ens_Zeidler','Enstatite','orangered'],
        ['En_Jaeger','Enstatite','orangered'],
        ['Ens_Jaeger','Enstatite','orangered'],
        ['_cor_','Corundum','fuchsia'],
        ['corundum','Corundum','fuchsia'],
        ['Am_Al2O3','Am. Al$_2$O$_3$','crimson'],
        ['hibonite', 'Hibonite', 'orchid'],
        ['Am_Ca2Al2SiO7','Am. gehlenite','slategray'],
        ['Am_CaMgAlsil_Mu','Am. CaMgAl-silicate','darkseagreen'],
        ['Am_Ca2Mg0_5AlSi1_5O7','Am. CaMgAl-silicate','darkseagreen'],
        ['Am_Mg2AlSi2O7.5','Am. MgAl-silicate','cadetblue'],
        ['Am_Mg3AlSi3O10','Am. MgAl-silicate','cadetblue'],
        ['_iron_','Iron','darkviolet'],
        ['_c-z_','Am. carbon','gray'],
        ['_sic_','SiC','rosybrown'], 
        ['Forsterite','Forsterite', 'lime'],
        ['Fayalite','Fayalite','deepskyblue'],
        ['Enstatite','Enstatite','orangered'],
        ['Olivine','Am. olivine','royalblue'],
        ['Pyroxene','Am. pyroxene','gold'],
        ['Am_Silica',r'Am. SiO$_2$','cyan'],
        ['Am_Silica_Kit',r'Am. SiO$_2$','cyan'],
        ['Silica_MH',r'Am. SiO$_2$','cyan'],
        ['Ann_Silica',r'Ann. SiO$_2$','magenta'],
        ['cristobalite',r'Cristobalite','magenta'],
        ['Silica',r'Am. SiO$_2$','cyan'],
        ['qua',r'SiO$_2$','cyan']
        ]
    
    gs_table=[\
        ['rv0.1',0.1],
        ['rv1.0',1.0],
        ['rv1.5',1.5],
        ['rv2.0',2.0],
        ['rv3.0',3.0],
        ['rv4.0',4.0],
        ['rv5.0',5.0],
        ['rv6.0',6.0],
        ['0.10um',0.1],
        ['1.00um',1.0],
        ['1.50um',1.5],
        ['2.00um',2.0],
        ['3.00um',3.0],
        ['4.00um',4.0],
        ['5.00um',5.0],
        ['6.00um',6.0],
        ['0.1um',0.1],
        ['1.0um',1.0],
        ['1.5um',1.5],
        ['2.0um',2.0],
        ['3.0um',3.0],
        ['4.0um',4.0],
        ['5.0um',5.0],
        ['6.0um',6.0],    
        ['0.1.',0.1],
        ['1.0.',1.0],
        ['1.5.',1.5],
        ['2.0.',2.0],
        ['3.0.',3.0],
        ['4.0.',4.0],
        ['5.0.',5.0],
        ['6.0.',6.0],  
        ['cristobalite',0.1],     
        ]
    color = 'black'
    short_label = kappa_label
    gs = 0.0
    for item in color_table:
        if item[0] in kappa_label:
            color = item[2]
            short_label = item[1]
            break
    for item in gs_table:
        if item[0] in kappa_label:
            gs = item[1]
            break
    return color,short_label,gs

#def get_short_label(kappa_label):
    #new_label = kappa_label.replace('Q_','').replace('DHS_','').replace('GRF_','').replace('.GRF','').replace('f0.7_','').replace('f0.8_','').replace('f0.99_','').replace('dhs_0.70','').replace('dhs_0.99','').replace('DHS_0.70','').replace('DHS_0.80','').replace('DHS_0.99','').replace('f1.0_','').replace('_rv','')
    #new_label = new_label.replace('Fo_Zeidler','Forsterite').replace('En_Zeidler','Enstatite').replace('En_Jaeger','Enstatite').replace('_Jae','').replace('_Dor','').replace('_MH','').replace('.Combined','')

def plot_fit(output_path,wl,fluxdata,fluxerr,model_fluxes,kappa_label_list,plot_title,
             param_dic,fit_mode,xlim=[],wl_filter=[],leg_loc='best',leg_ncol=3,
             axr_ylim=[np.nan,np.nan]):
    #fig, ((ax)) = plt.subplots(1, 1, sharey=False, sharex=False,figsize=(8,6))
    fig = plt.figure(figsize=(8,6.5)) #width, height
    gs1 = GridSpec(2, 1, height_ratios=[3.5,1],bottom=0.35,top=0.94,hspace=0.02)
    gs2 = GridSpec(1, 1,top=0.27)
    ax = fig.add_subplot(gs1[0])
    axr = fig.add_subplot(gs1[1])
    axb = fig.add_subplot(gs2[0])

    #if 'AV' in param_dic:
    #    #experimental: de-extinct the data
    #    flux_new = fluxdata/ext.extinguish(wl*10000*u.AA, Av=param_dic['AV']['value'])
    #    fluxdata = flux_new 
    
    l0, = ax.plot(wl,fluxdata,'-',label='Data',lw=2,color='black',zorder=1.4) #'+'
    l05 = ax.fill_between(wl, fluxdata-fluxerr, fluxdata+fluxerr, 
        facecolor=(0,0,0,0.2),edgecolor=(0,0,0,0),zorder=1.39)
    model_flux,flux_dust,flux_midplane,flux_star,flux_rim,dust_comp_flux = model_fluxes
    dust_comp_line_lst = []
    for i,kappa_label in enumerate(kappa_label_list):
        linestyle = get_linestyle(kappa_label)
        linecolor = (get_color_label(kappa_label))[0]
        line, = ax.plot(wl,dust_comp_flux[:,i],linestyle=linestyle,color=linecolor,alpha=0.8,label=(get_color_label(kappa_label))[1],lw=1,zorder=1.2)
        dust_comp_line_lst.append(line)
    l1, = ax.plot(wl,flux_dust,'-b',label='Dust',lw=1.5,zorder=1.35)
    l2, = ax.plot(wl,flux_midplane,'-g',label='Midplane',lw=1.5,zorder=1.3)
    l3, = ax.plot(wl,flux_rim,'-',label='Rim',lw=1.5,color='purple',zorder=1.3)
    l4, = ax.plot(wl,flux_star,'-',label='Star',lw=1.5,color='orange',zorder=1.3)
    l5, = ax.plot(wl,model_flux,'-r',label='Model',lw=2,alpha=0.7,zorder=1.5)
    #ax.set_ylim((ymin,ymax))
    #ax.set_xlabel('Wavelength ($\mu$m)')
    ax.set_ylabel('Flux density (Jy)')
    #apply wl filter
    for wl_range in wl_filter:
        #fl_filt = np.interp(wl_range, wl, fluxdata) 
        #l0m, = ax.plot(wl_range,fl_filt,'-',label='_nolegend_',lw=3,color='white')
        wl_lim = ax.get_xlim()
        ylim = ax.get_ylim()
        if not(wl_lim[0] > wl_range[1] or wl_range[0] > wl_lim[1]):
            rect = plt.Rectangle((wl_range[0], ylim[0]), wl_range[1]-wl_range[0], ylim[1]-ylim[0],
                    facecolor="white",zorder=1.9)
            ax.add_patch(rect) 
    # ax.grid(which='minor',alpha=0.3)
    # ax.grid(which='major')
    ax.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="minor",zorder=10.0)
    ax.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="major",zorder=10.0)
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.minorticks_on( )
    ax.tick_params('x', labelbottom=False,zorder=10.0)
    ax.set_axisbelow(False)
    if len(xlim) == 2:
        ax.set_xlim(xlim)

    # ax.tick_params(axis="x", which="minor", direction="in", 
    #                   top=True, bottom=True) #, labelbottom=False)
    plt.suptitle(plot_title)
    leg1 = ax.legend(handles=[l0,l5,l1,l2,l3,l4],framealpha=0.66,ncol=leg_ncol,loc=leg_loc,
        handletextpad=0.4,handlelength=2.0,prop={'size': 11.5},
        columnspacing=0.6) #.set_zorder(51) #loc='upper right'

    axr.plot(wl,0.0*wl,'--',color='gray',lw=1.5,zorder=2)
    residual = 100.0*(fluxdata-model_flux)/fluxdata
    axr.plot(wl,residual,'-k',label='Model',lw=2,zorder=1.4)
    axr.fill_between(wl, residual-100.0*fluxerr/fluxdata, residual+100.0*fluxerr/fluxdata, 
        facecolor=(0,0,0,0.2),edgecolor=(0,0,0,0),zorder=1.39)
    #for wl_range in wl_filter:
    #    fl_filt = np.interp(wl_range, wl, residual) 
    #    axr.plot(wl_range,fl_filt,'-',label='_nolegend_',lw=3,color='white')
    #apply wl filter
    for wl_range in wl_filter:
        wl_lim = axr.get_xlim()
        ylim = axr.get_ylim()
        if not(wl_lim[0] > wl_range[1] or wl_range[0] > wl_lim[1]):
            rect = plt.Rectangle((wl_range[0], ylim[0]), wl_range[1]-wl_range[0], ylim[1]-ylim[0],
                    facecolor="white",zorder=1.9)
            axr.add_patch(rect)     
    axr.set_xlabel(r'Wavelength ($\mu$m)')
    axr.set_ylabel('Residual (%)')
    axr.xaxis.set_major_locator(loc)
    axr.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="minor",zorder=10)
    axr.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="major",zorder=10)
    axr.minorticks_on( )
    axr.set_axisbelow(False)
    if len(xlim) == 2:
        axr.set_xlim(xlim)
    if ~np.isnan(axr_ylim[0]):
        axr.set_ylim(bottom=axr_ylim[0])
    if ~np.isnan(axr_ylim[1]):
        axr.set_ylim(top=axr_ylim[1])

    #leg2 = axr.legend(handles=dust_comp_line_lst,loc='lower center',bbox_to_anchor=(0.5, -0.07),
    #                bbox_transform=fig.transFigure, ncol=3,fontsize=12)
    
    #bar chart
    surfdens = 0.0
    for kappa_label in kappa_label_list:
        if fit_mode == 'full':
            surfdens += 10.0**param_dic[kappa_label]['value']
        elif fit_mode == 'with_nnls':
            surfdens += param_dic[kappa_label]['value']
    xlabels = []
    yvalues = []
    linestyle_lst = []
    barcolor_lst = []
    for kappa_label in kappa_label_list: 
        # tmp = param_dic[kappa_label]['grain_size']
        # if '.0' and tmp.endswith('.0'):
        #     tmp = tmp[:-len('.0')]
        if param_dic[kappa_label]['grain_size']<1.0:
            xlabels.append('%.1f'%(param_dic[kappa_label]['grain_size']))
        else:
            xlabels.append('%.0f'%(param_dic[kappa_label]['grain_size']))
        yvalues.append(param_dic[kappa_label]['value']/surfdens)
        linestyle_lst.append(get_linestyle(kappa_label))
        barcolor_lst.append((get_color_label(kappa_label))[0])
    bar_width=0.8
    axb.bar(range(len(xlabels)), yvalues,tick_label=xlabels,width=bar_width,
            log=True,color=barcolor_lst,alpha=1.0)#,edgecolor='black') # label=xl, color=bar_colors)
    ymin = 0.01
    for i in range(len(yvalues)):
        if yvalues[i] <= ymin:
            height = np.nan
        else:
            height = yvalues[i]-ymin

        axb.add_patch(Rectangle((i-bar_width/2.0, ymin), bar_width, height,
        fill=False,edgecolor='black', transform=axb.transData, clip_on=True,
        linestyle=linestyle_lst[i],linewidth=1.0))
    prev_name = ''
    first = True
    for i,(kappa_label) in enumerate(kappa_label_list):
        name = param_dic[kappa_label]['grain_name']
        name = (get_color_label(name))[1]

        if prev_name != name:
            axb.annotate(name,(i-bar_width/2.0,0.5),size=11)
            if first == True:
                first = False
            else:
               axb.plot([i-0.5,i-0.5],[ymin,1.0],'--',color='lightgray')
        prev_name = name

    axb.set_ylim(ymin,1.0)
    axb.set_ylabel('Mass fraction')
    axb.set_xlabel(r'Grain size ($\mu$m)')
    # print(xlabels, yvalues)
    #print(output_path)
    plt.tight_layout() #pad=0.5)
    if 'pdf' in str(output_path):
        plt.savefig(output_path, dpi=200,bbox_inches="tight")
    else:
        plt.savefig(output_path, dpi=200,bbox_inches="tight")
    #plt.show()

def read_fit_param_file(best_params_file_path,wl):
    param_dic = {}
    kappa_label_list = []    
    free_param_start = False
    fix_param_start = False
    dust_param_start = False
    opac_flst_start = False
    chi2r = np.inf
    opac_fname_list = []
    grain_name_list = []
    grain_size_list = []
    with open(best_params_file_path, "r") as f:
        for line in f:
            if line.startswith('--------'):
                free_param_start = False
                fix_param_start = False
                dust_param_start = False
            if line.startswith('Chi2r'):
                chi2r = float((line.split(':'))[1])
            if free_param_start:
                # print((line.split())[0].replace(':',''))
                # print(float((line.split())[1]))
                # print(((line.split())[2]))
                param_dic[(line.split())[0].replace(':','')] = {'value':float((line.split())[1]),'limits':[np.nan,np.nan],'free':True}
            if fix_param_start:
                # print((line.split())[0].replace(':',''))
                # print(float((line.split())[1]))
                label = (line.split())[0].replace(':','')
                # gs = re.sub("[^\d\.]", "",get_short_label(label))
                # gn = re.sub(r'[0-9.]+', '', get_short_label(label))
                gn = (get_color_label(label))[1]
                gs = (get_color_label(label))[2]
                param_dic[label] = {'value':float((line.split())[1]),'limits':[np.nan,np.nan],'free':False,'grain_size':gs,'grain_name':gn}
            if dust_param_start:
                label = (line.split())[0].replace(':','')
                kappa_label_list.append(label)
                #kappa_label_short_list.append((get_color_label(label))[1])
                grain_name_list.append((get_color_label(label))[1])
                grain_size_list.append((get_color_label(label))[2])
                param_dic[label]={'value':5e-4,'limits':[0.0,5e-2],'free':False, 
                                'grain_name':grain_name_list[-1] ,'grain_size':grain_size_list[-1],'mass_fraction':np.nan}
            if opac_flst_start:
                opac_fname_list.append( line.strip() )
            if line.startswith('Free parameters'):
                free_param_start = True
            if line.startswith('Fixed parameters'):
                fix_param_start = True
            if line.startswith('Dust mass fractions'):
                dust_param_start = True
            if line.startswith('Input opacity files'):
                opac_flst_start = True
        
        if not(not wl):
            freq = (c0*1e6)/wl
        else:
            freq = (c0*1e6)/np.array([1.0,10.0])
        param_dic['n_r']['value'] = int(param_dic['n_r']['value'])
        param_dic['n_rim']['value'] = int(param_dic['n_rim']['value'])
        #precomputed dimensionless radial grids
        x = np.linspace(0.0, 1.0, param_dic['n_r']['value'])
        x_rim = np.linspace(0.0, 1.0,  param_dic['n_rim']['value'])
        param_dic['x'] =        {'value':x,'limits':[0.0,1.0],'free':False}
        param_dic['x_rim'] =        {'value':x_rim,'limits':[0.0,1.0],'free':False}

        #precomputed stellar spectrum
        scale = 1e26 / param_dic['dscale']['value']**2
        param_dic['scale'] = {'value':scale,'limits':[-1.0,1.0],'free':False}
        Bstar = B_nu_scalar(param_dic['T_star']['value'], freq)
        flux_star = scale * (param_dic['r_star']['value']*rsun/au)**2 * np.pi * Bstar
        param_dic['flux_star'] = {'value':flux_star,'limits':[0.0,1.0],'free':False}

    return param_dic,kappa_label_list,opac_fname_list,chi2r

# opac_type =  'Q_abs' or 'kappa_abs'
def read_kappa_arr(opac_dir,opac_fname_list,opac_type,wl):
    kappa_arr = np.zeros((len(wl),len(opac_fname_list)))
    for i,fname in enumerate(opac_fname_list):
        print(fname,end=' ')
        if opac_type == 'Q_abs':
            N,gsize,gdens=np.loadtxt(opac_dir+'/'+fname,max_rows=1)
        n_comment_lines = 0
        with open(opac_dir+'/'+fname) as f:
            for line in f:
                if line.startswith('#'):
                    n_comment_lines += 1
                else:
                    break
        if opac_type == 'Q_abs':
            wl_Q,opac_data = np.loadtxt(opac_dir+'/'+fname, comments="#", skiprows=n_comment_lines+2, usecols=(0,1), unpack=True)
            kappa_arr[:,i] = np.interp(wl, wl_Q, 0.1* opac_data* 3.0 / (4.0 * gsize * 1e-4 * gdens)) #m^2/kg
        if opac_type == 'kappa_abs':
            # GRF opacity files: lambda[um], kappa_tot, kappa_abs, kappa_sca
            # optool opacity files: lambda[um]  kabs [cm^2/g]  ksca [cm^2/g]    g_asymmetry
            if n_comment_lines > 20:
                usec = (0,1) # optool opacity file
                print('optool opacity file')
            else: 
                usec = (0,2) #GRF opacity files
                print('GRF opacity file')
            wl_Q,opac_data = np.loadtxt(opac_dir+'/'+fname, comments="#", skiprows=n_comment_lines+2, usecols=usec, unpack=True)
            kappa_arr[:,i] = np.interp(wl, wl_Q, 0.1* opac_data) #convert to m^2/kg
    return kappa_arr

# %%

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx