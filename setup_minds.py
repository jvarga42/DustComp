#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class OpacitySetup:
    opac_fname_list: list
    opac_dir: Path
    opac_type: str #'Q_abs' or 'kappa_abs'
    name: str = "custom"

# ----------------------------------------------------------------------
# Instructions
#   1) Create your own setup_...py file, or adapt this file for your
#      needs.
#   2) In the setup file, specify the following:
#          - fitter options
#          - input data
#          - opacity files
#   3) In run_DustComp.py, set the SETUP_FILE variable to the path
#      of your setup file.
#   4) Run the code: python3 run_DustComp.py
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Setup file for the MINDS survey of silicates in T Tauri disks
# Varga et al. 2026
# https://ui.adsabs.harvard.edu/abs/2026A%26A...711A.125V/abstract
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# User-editable fitter options
# ----------------------------------------------------------------------

fit_method = 'dynesty' #DO NOT CHANGE IT. 'mcmc' was supported previously, but not anymore
maxiter = 200000 #maximum number of iterations if you use dynesty
# fit_mode:
#  'full': all parameters are fitted with Dynesty, 
#  'with_nnls': the abundance coefficients are derived with a least square algorithm, the rest are fitted with dynesty
fit_mode = 'with_nnls' #'full', 'with_nnls' is the preferred mode

fit_two_zones = False # if True, the model has two radial zones with different compositions

# If you want to perform a fit, set do_fit to True 
do_fit = True
do_plot_init = False
# If you want to postprocess earlier fit results, set load_previous_results to True (and do_fit to False). You should also set plot_previous_results and calc_errors.
load_previous_results = False
plot_previous_results = False
calc_errors = False # to calculate uncertainties of the mass fractions

# set your path to DustComp here
DC_dir = '/Users/jvarga/Dokumentumok/MATISSE/pro/DustComp/'

wl_limits =  [4.9,27.0]  #[7.5,13.5] #um

# ----------------------------------------------------------------------
# Select input data to fit
# ----------------------------------------------------------------------

datadir = Path(DC_dir) / 'example_data'
outputdir = Path(DC_dir) / 'example_results' 

# spec_data is a list where each item is also list containing the following:
# filepath, distance(pc), T_eff(K), L_star(L_Sun), r_star(R_Sun),output folder,list of wavelength ranges to be excluded from the fit
# r_star can be omitted (np.nan) if L_star is known
spec_data =[\
[datadir / 'XXCha_smoothed_rebinned.dat', 194.64,3340.0,0.29,np.nan,outputdir,[ [4.8,5.2],[6.4,6.9],[13.3,14.2],[14.8,15.1],[25.5,35.0]] ] ,
]

[
[datadir / 'VWChaA_smoothed_rebinned.dat', 188.16,4060.0,2.21,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[7.4,7.6],[13.3,14.2],[14.8,15.1],[26.5,35.0]] ],
[datadir / 'IMLup_smoothed_rebinned.dat',153.81,4350.0,2.57,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[13.3,14.2],[14.8,15.1],[25.5,35.0]] ],
[datadir / 'DFTau_smoothed_rebinned.dat',176.45,3900.0,3.89,np.nan,outputdir,[[4.8,5.2],[6.2,7.2],[13.3,14.2],[24.4,25.4]] ],
[datadir / 'Sz50_smoothed_rebinned.dat',  148.94,3400.0,0.41,np.nan,outputdir,[[6.4,6.9],[13.5,14.2]] ], #dist Bailer-Jones, others: Sartori 2003
[datadir / 'WAOph6_smoothed_rebinned.dat',122.53,4169.0,2.88,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[13.3,14.2],[14.8,15.1],[25.5,35.0]] ], #Andrews+2018 DSHARP
[datadir / 'GWLup_smoothed_rebinned.dat',155.20,3632.0,0.33,np.nan,outputdir,[[4.8,5.2],[6.5,6.9],[13.5,14.2],[25.5,35.0]] ],
[datadir / 'DNTau_smoothed_rebinned.dat',127.29,3806.0,0.69,np.nan,outputdir,[[6.8,6.9],[13.3,14.2]] ],
[datadir / 'CXTau_smoothed_rebinned.dat',126.74,3487.0,0.34,np.nan,outputdir,[[6.4,6.9],[13.3,14.2]] ],
[datadir / 'BPTau_smoothed_rebinned.dat',128.28,3777.0,0.83,np.nan,outputdir,[[4.8,5.2],[6.4,6.7],[25.0,35.0]] ],
[datadir / 'CYTau_smoothed_rebinned.dat',124.35,3516.0,0.37,np.nan,outputdir,[[6.4,6.7],[13.4,14.2],[14.8,15.1],[24.0,35.0]] ],
[datadir / 'DLTau_smoothed_rebinned.dat',159.53,4276.0,1.47,np.nan,outputdir,[[4.8,5.2],[6.5,6.9],[13.3,14.2],[25.5,26.5]] ],
[datadir / 'DMTau_smoothed_rebinned.dat',144.8, 3715.0,0.16,np.nan,outputdir,[] ], #Lopez-Martinez+2015
[datadir / 'AATau_smoothed_rebinned.dat',137.72,3762.0,0.72,np.nan,outputdir,[[13.5,14.2],[14.8,15.1],[24.8,26.0]] ],
[datadir / 'DRTau_smoothed_rebinned.dat',186.98,4202.0,3.71,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[13.3,14.2],[14.9,15.05],[25.5,35.0]] ],
[datadir / 'FTTau_smoothed_rebinned.dat',129.96,3415.0,0.44,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[7.4,7.6],[13.3,14.2],[14.9,15.05],[25.6,35.0]] ],
[datadir / 'LkCa15_smoothed_rebinned.dat',154.83,4276.0,1.12,np.nan,outputdir,[] ],
[datadir / 'PDS70_smoothed_rebinned.dat', 112.32,4138.0,0.38,np.nan,outputdir,[[26.6,35.0]] ],
[datadir / 'RNO90_smoothed_rebinned.dat', 114.96,5662.0,5.7 ,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[7.4,7.6],[13.5,14.2],[14.8,15.1],[25.5,35.0]] ], #Salyk+2013
[datadir / 'RWAurA_smoothed_rebinned.dat', 150.0 ,4870.0,0.83,np.nan,outputdir,[[4.8,5.2],[5.9,6.9],[14.8,15.1],[25.5,35.0]] ], #Testi+2022
[datadir / 'RWAurB_smoothed_rebinned.dat', 150.0 ,4160.0,0.50,np.nan,outputdir,[[4.8,5.2],[5.9,6.9],[14.8,15.1],[25.5,35.0]] ], #Testi+2022
[datadir / 'SYCha_smoothed_rebinned.dat', 180.78,4060.0,0.43,np.nan,outputdir,[[6.4,6.9],[13.5,14.2],[26.0,35.0]] ],
[datadir / 'Sz98_smoothed_rebinned.dat',  156.27,4060.0,1.51,np.nan,outputdir,[[6.4,6.9],[13.7,14.2],[26.2,35.0]] ],
[datadir / 'TWHya_smoothed_rebinned.dat',  59.96,4000.0,0.34,np.nan,outputdir,[[7.4,7.6],[26.5,35.0]] ],
[datadir / 'V1094Sco_smoothed_rebinned.dat',152.44,4205.0,1.15,np.nan,outputdir,[[6.4,6.9],[13.3,14.2],[23.8,35.0]] ],
[datadir / 'WXChaA_smoothed_rebinned.dat', 189.3 ,3710.0,1.13,np.nan,outputdir,[[4.8,5.2],[6.4,6.9],[13.3,14.2],[14.8,15.1],[25.5,35.0]] ], #Testi+2022
]   

# ----------------------------------------------------------------------
# Select opacities
# ----------------------------------------------------------------------

# Either you specify your own selection (custom setup), or you use the run_name-based opacity setup.
# In the latter approach, run names have a specific syntax which is used to devise the grain opacities used in the fitting.
# The names provided here are the ones used in the Varga+2026 MINDS silicate survey paper.
# Here are some options in the run name syntax:
# - opacity set: 
#     newDHS: corresponds to the DHS_nat set in the Varga+2026 paper
#     oldDHS: corresponds to the DHS_synth set in the Varga+2026 paper 
#     GRF: GRF set of opacities
#     custom_mix: custom set given in the opac_fname_list_custom_mix list
# - incluion of SiO2:
#     no_SiO2: no SiO2 opacities used
#     no_ann_SiO2: only amorphous silica opacities used
#     with_ann_SiO2: both amorphous and annealed silica opacities used
# - grain sizes (um):
#     2gs: [0.1,2.0] 
#     3gs: [0.1,2.0,5.0]
#     3vgs: [0.1,1.0,2.0]
#     4gs: [0.1,1.0,2.0,5.0]
#     6gs: [0.1,1.0,2.0,3.0,4.0,5.0] 
# - modifiers:
#     nobigcrystals: 5 um-sized crystalline grains not included
#     ...

opacity_setup_type = 'run_name' #'custom'
 
# a) If opacity_setup_type is 'run_name':
# =======================================
run_names = [
'36_newDHS_Qtemp_300K_with_ann_SiO2_nobigcrystals_3gs',
'37_oldDHS_with_ann_SiO2_nobigcrystals_3gs',
'38_GRF_with_ann_SiO2_nobigcrystals_3gs',
'39_newDHS_Qtemp_300K_no_ann_SiO2_nobigcrystals_3gs',
'40_oldDHS_no_ann_SiO2_nobigcrystals_3gs',
'41_GRF_no_ann_SiO2_nobigcrystals_3gs'
]

# b) If opacity_setup_type is 'custom':
# =======================================

# run_names = ['42_DHS_with_fors_aerosol']
custom_setup = OpacitySetup(
    opac_fname_list=[
        'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',      
        'Q_Am_Mgol_Jae_DHS_f0.70_rv2.0.dat',      
        'Q_Am_Mgol_Jae_DHS_f0.70_rv5.0.dat',      
        #'Q_Fo_Zeidler_300K_DHS_f0.99_rv0.1.dat',  
        #'Q_Fo_Zeidler_300K_DHS_f0.99_rv2.0.dat',  
        'Q_Fors_aerosol.dat',
        'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',     
        'Q_Am_Mgpyr_Dor_DHS_f0.70_rv2.0.dat',     
        'Q_Am_Mgpyr_Dor_DHS_f0.70_rv5.0.dat',     
        'Q_Ens_Zeidler_300K_DHS_f0.99_rv0.1.dat', 
        'Q_Ens_Zeidler_300K_DHS_f0.99_rv2.0.dat', 
        'Q_Am_Silica_Kit_DHS_f0.70_rv0.1.dat',    
        'Q_Am_Silica_Kit_DHS_f0.70_rv2.0.dat',    
        'Q_Am_Silica_Kit_DHS_f0.70_rv5.0.dat',    
        'Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat',
        'Q_Ann_Silica_Fabian_DHS_f0.99_rv2.0.dat',
    ],
    opac_dir=Path(DC_dir) / 'opacities' /'QVAL',
    opac_type='Q_abs',
    name="custom",
)

def get_opacity_setup(run_name=None,
                      custom_setup=None,
                      DC_dir=None):
    """
    Returns an OpacitySetup object.

    Exactly one of
        run_name
        custom_setup
    must be supplied.

    Parameters
    ----------
    run_name : str
        Name of a predefined opacity setup.

    custom_setup : OpacitySetup
        A user-defined opacity setup.

    DC_dir : str or Path
        DustComp directory.
    """

    if (run_name is None) == (custom_setup is None):
        raise ValueError(
            "Specify exactly one of run_name or custom_setup."
        )

    # -------------------------------------------------------------
    # User supplied everything directly
    # -------------------------------------------------------------

    if custom_setup is not None:
        return custom_setup

    # -------------------------------------------------------------
    # Existing run_name logic
    # -------------------------------------------------------------

    DC_dir = Path(DC_dir)
    opac_dir_GRF = DC_dir / 'opacities' / 'kappa_abs' / 'GRF_opacity'
    opac_dir_DHS = DC_dir / 'opacities' / 'QVAL' 

    if 'DHS' in run_name:
        opac_dir = opac_dir_DHS
        opac_type =  'Q_abs' 
    if 'GRF' in run_name:
        opac_dir = opac_dir_GRF
        opac_type =  'kappa_abs'  

    gsize_list = []
    if '6gs' in run_name:
        gsize_list = [0.1,1.0,2.0,3.0,4.0,5.0]   
    elif '4gs' in run_name:
        gsize_list = [0.1,1.0,2.0,5.0]    
    elif '3gs' in run_name:
        gsize_list = [0.1,2.0,5.0]   
    elif '3vgs' in run_name:
        gsize_list = [0.1,1.0,2.0] 
    elif '2gs' in run_name:
        gsize_list = [0.1,2.0]  

    opac_species_list = []   
    if 'GRF' in run_name:
        if 'FeGRF' in run_name:
            opac_species_list = [\
            'MgOlivine0.1.Combined.Kappa',
            'Olivine0.1.Combined.Kappa',
            'Forsterite0.1.Combined.Kappa',
            'Fayalite0.1.Combined.Kappa',
            'MgPyroxene0.1.Combined.Kappa',
            'Pyroxene0.1.Combined.Kappa',
            'Enstatite0.1.Combined.Kappa',
            ]
        elif 'MgGRF' in run_name:
            opac_species_list= [\
            'MgOlivine0.1.Combined.Kappa',
            #'Olivine0.1.Combined.Kappa',
            'Forsterite0.1.Combined.Kappa',
            #'Fayalite0.1.Combined.Kappa',
            #'MgPyroxene0.1.Combined.Kappa',
            #'Pyroxene0.1.Combined.Kappa',
            #'Enstatite0.1.Combined.Kappa',
            ]
        else:
            opac_species_list = [\
            'MgOlivine0.1.Combined.Kappa',
            ##'Olivine0.1.Combined.Kappa',
            'Forsterite0.1.Combined.Kappa',
            'MgPyroxene0.1.Combined.Kappa',
            ##'Pyroxene0.1.Combined.Kappa',
            'Enstatite0.1.Combined.Kappa',
            ##'Fayalite0.1.Combined.Kappa',
            ]
        if not('no_SiO2' in run_name):
            opac_species_list.append('Silica0.1.Combined.Kappa')
            if not('no_ann_SiO2' in run_name):
                opac_species_list.append('kappa_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat')
        if 'Al' in run_name:
            opac_species_list.append('kappa_Am_Mg3AlSi3O10.5_Mutschke1998_DHS_f0.70_rv0.1.dat')
        if 'Ca' in run_name:
            opac_species_list.append('kappa_Am_Ca2Al2SiO7_Mutschke_DHS_f0.70_rv0.1.dat')
        if 'with_corundum' in run_name:
            opac_species_list.append('kappa_Am_Al2O3_compact_Begemann1997_DHS_f0.70_rv0.1.dat')
        
        opac_fname_list=[]
        for species in opac_species_list:
            if 'nobigcrystals' in run_name:
                if '_Fo' in species or '_Ens' in species or '_Ann_Silica' in species or 'Forsterite' in species or 'Enstatite' in species:
                    for gsize in gsize_list[:-1]:
                        opac_fname_list.append(species.replace('0.1','%.1f'%(gsize)))   
                else:
                    for gsize in gsize_list:
                        opac_fname_list.append(species.replace('0.1','%.1f'%(gsize))) 
            else:
                for gsize in gsize_list:
                    opac_fname_list.append(species.replace('0.1','%.1f'%(gsize)))
                    
    elif 'DHS' in run_name:
        if 'newDHS' in run_name:
            if 'Qtemp' in  run_name:
                qtemp = run_name.split('Qtemp_',1)[1].split('K',1)[0]
                opac_species_list = [\
                'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
                'Q_Fo_Zeidler_'+qtemp+'K_DHS_f0.99_rv0.1.dat',
                'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
                'Q_Ens_Zeidler_'+qtemp+'K_DHS_f0.99_rv0.1.dat',
                ]   
            else:
                if 'fmax' in run_name:
                    fmax = run_name.split('fmax',1)[1].split('_',1)[0]
                    opac_species_list = [\
                    'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
                    'Q_Fo_Zeidler_DHS_f%s_rv0.1.dat'%fmax,
                    'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
                    'Q_Ens_Zeidler_DHS_f%s_rv0.1.dat'%fmax,
                    ]  
                else:
                    opac_species_list = [\
                    'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
                    'Q_Fo_Zeidler_DHS_f0.99_rv0.1.dat',
                    'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
                    'Q_Ens_Zeidler_DHS_f0.99_rv0.1.dat',
                    ]   
        elif 'bestDHS' in run_name:
            opac_species_list = [\
            'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
            'Q_Fo_Zeidler_300K_DHS_f0.70_rv0.1.dat',
            'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
            'Q_Ens_Zeidler_300K_DHS_f0.70_rv0.1.dat',
            ]
        elif 'oldDHS' in run_name:  
            if 'fmax' in run_name:
                fmax = run_name.split('fmax',1)[1].split('_',1)[0]
                opac_species_list = [\
                'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
                'Q_Fo_Suto_DHS_f%s_rv0.1.dat'%fmax,
                'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
                'Q_Ens_Jaeger_DHS_f%s_rv0.1.dat'%fmax,
                ]   
            else:
                
                opac_species_list = [\
                'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
                'Q_Fo_Suto_DHS_f0.99_rv0.1.dat',
                'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
                'Q_Ens_Jaeger_DHS_f0.99_rv0.1.dat',
                ]   
        elif 'MgFeDHS' in run_name:   
            opac_species_list = [\
            'Q_Am_Mgol_Jae_DHS_f0.00_rv0.1.dat',
            'Q_Am_Ol-Mg50_MgFeSiO4_Dorschner_DHS_f0.00_rv0.1.dat',
            'Q_Fo_Servoin_GRF_rv0.1.dat', #from Jay
            'Q_Am_Mgpyr_Dor_DHS_f0.00_rv0.1.dat',
            'Q_Am_MgFepyr-Mg50_Dor_DHS_f0.00_rv0.1.dat',
            #'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
            #'Q_Am_Ol-Mg50_MgFeSiO4_Dorschner_DHS_f0.70_rv0.1.dat',
            #'Q_Fo_Servoin_GRF_rv0.1.dat', #from Jay
            #'Q_Fo_Suto_DHS_f0.99_rv0.1.dat', #'Q_Fay_Fabian_DHS_f0.99_rv0.1.dat', #, #'Q_Fo_Zeidler_DHS_f0.99_rv0.1.dat', # # #
            #'Q_Fay_Fabian_DHS_f0.99_rv0.1.dat',
            #'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
            #'Q_Am_MgFepyroxene_Dor_DHS_f0.7_rv0.1.dat',
            'Q_Ens_Jaeger_DHS_f0.99_rv0.1.dat', #'Q_Ens_Zeidler_DHS_f0.99_rv0.1.dat',
            #'Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat',
            ]
        elif 'MgDHS' in run_name:   
            opac_species_list = [\
            'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
            'Q_Fo_Servoin_GRF_rv0.1.dat', #from Jay
            #'Q_Fo_Suto_DHS_f0.99_rv0.1.dat', #'Q_Fay_Fabian_DHS_f0.99_rv0.1.dat', #, #'Q_Fo_Zeidler_DHS_f0.99_rv0.1.dat', # # #
            #'Q_Fay_Fabian_DHS_f0.99_rv0.1.dat',
            #'Q_Am_MgFepyroxene_Dor_DHS_f0.7_rv0.1.dat',
            #'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
            #'Q_Ens_Jaeger_DHS_f0.99_rv0.1.dat', #'Q_Ens_Zeidler_DHS_f0.99_rv0.1.dat',
            #'Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat',
            ]
        elif 'AlDHS' in run_name:
            #opac_species_list = [\
            #'Q_Am_Mg2AlSi2O7.5_Mutschke1998_DHS_f0.70_rv0.1.dat',
            #'Q_Fo_Zeidler_300K_DHS_f0.70_rv0.1.dat',
            #'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
            #]
            opac_species_list = [\
            'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
                #'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
            #'Q_Ens_Zeidler_300K_DHS_f0.99_rv0.1.dat',
            #'Q_Am_Mg2AlSi2O7.5_Mutschke1998_DHS_f0.70_rv0.1.dat',
            'Q_Am_Mg3AlSi3O10.5_Mutschke1998_DHS_f0.70_rv0.1.dat',
            'Q_Am_Ca2Al2SiO7_Mutschke_DHS_f0.70_rv0.1.dat',
            #'Q_Am_Ca2Mg0_5AlSi1_5O7_Mutschke_DHS_f0.70_rv0.1.dat',
            #'Q_Koike_Fo100_combined.dat',
            #'Q_Forsterite0.1.GRF.dat',
            'Q_Fo_Servoin_GRF_rv0.1.dat', #from Jay
            #'Q_Fo_Zeidler_300K_DHS_f0.99_rv0.1.dat',
            #'Q_Fo_Suto_DHS_f0.99_rv0.1.dat',
            #'Q_Fors_aerosol.dat',
            #'Q_hibonite_Mutschke2002_DHS_f0.99_rv0.1.dat'
            ]
            
        if not('no_SiO2' in run_name):
            opac_species_list.append('Q_Am_Silica_Kit_DHS_f0.70_rv0.1.dat')     
            #opac_species_list.append('Q_Am_Silica_HM1997_300K_DHS_f0.70_rv0.1.dat')    
            if not('no_ann_SiO2' in run_name):
                    if 'cristobalite_Koike' in run_name:
                        opac_species_list.append('Q_alpha_cristobalite_Koike2013_rv0.02.dat')
                    elif 'with_cristobalite' in run_name:
                        opac_species_list.append('Q_cristobalite.dat')
                    else:
                        opac_species_list.append('Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat') 
        if 'Al' in run_name:
            opac_species_list.append('Q_Am_Mg3AlSi3O10.5_Mutschke1998_DHS_f0.70_rv0.1.dat')
        if 'Ca' in run_name:
            opac_species_list.append('Q_Am_Ca2Al2SiO7_Mutschke_DHS_f0.70_rv0.1.dat')
        if 'with_corundum' in run_name:
            opac_species_list.append('Q_Am_Al2O3_compact_Begemann1997_DHS_f0.70_rv0.1.dat')
            
        if 'DHS' in run_name:         
            opac_fname_list=[]
            for species in opac_species_list:
                if 'cristobalite' in species:
                    opac_fname_list.append(species)
                elif 'aerosol' in species:
                    opac_fname_list.append(species)
                elif 'combined' in species:
                    opac_fname_list.append(species)
                else:
                    if 'nobigcrystals' in run_name:
                        if '_Fo' in species or '_Ens' in species or '_Ann_Silica' in species or 'Forsterite' in species or 'Enstatite' in species:
                            for gsize in gsize_list[:-1]:
                                opac_fname_list.append(species.replace('0.1','%.1f'%(gsize)))   
                        else:
                            for gsize in gsize_list:
                                opac_fname_list.append(species.replace('0.1','%.1f'%(gsize))) 
                    else:
                        for gsize in gsize_list:
                            opac_fname_list.append(species.replace('0.1','%.1f'%(gsize)))

    return OpacitySetup(
        opac_fname_list=opac_fname_list,
        opac_dir=opac_dir,
        opac_type=opac_type,
        name=run_name,
    )



                
        
