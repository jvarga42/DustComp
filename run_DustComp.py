#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
#%%
from pathlib import Path
import importlib.util

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Set the path of the setup file module here
SETUP_FILE = Path("setup_minds.py")
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

try:
    SETUP_FILE
except NameError:
    SETUP_FILE = Path("setup_minds.py")
spec = importlib.util.spec_from_file_location(
    "setup",
    SETUP_FILE
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

from DustCompLib import * 
import numpy as np
# import emcee
import corner
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import multiprocessing as mp
import os
# from scipy.optimize import nnls
# from scipy import stats
# import re
import time
import pickle

os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

for run_name in setup.run_names:
    print(run_name)
    t0 = time.perf_counter()

    if setup.opacity_setup_type == 'run_name':  
        opac_setup = setup.get_opacity_setup(
            run_name=run_name,
            DC_dir=setup.DC_dir,
        )
    elif setup.opacity_setup_type == 'custom':
        opac_setup = setup.get_opacity_setup(custom_setup=setup.custom_setup)
    opac_fname_list = opac_setup.opac_fname_list
    opac_type = opac_setup.opac_type
    opac_dir = opac_setup.opac_dir

    spec_path_lst = []
    distance_lst = []
    T_star_lst = [] #K
    r_star_lst = [] #R_Sun
    lum_star_lst = [] #L_Sun
    output_path_lst = []
    wl_filter_lst = []
    wl_lst = []
    freq_lst = []
    fluxdata_lst = []
    kappa_arr_lst = []
    
    for item in setup.spec_data:
        spec_path_lst.append(item[0])
        distance_lst.append(item[1])
        T_eff = item[2]
        Lum = item[3]
        T_star_lst.append(T_eff)
        lum_star_lst.append(Lum)
        try:
            r_star = np.sqrt(Lum * 3.828e26 / (4.0*np.pi*sigma_SB*T_eff**4))/7.1492e7/9.96
        except ZeroDivisionError as e:
            r_star = 0.001
        r_star_lst.append(r_star)
        output_path_lst.append(item[5])
        wl_filter_lst.append(item[6])
        # print(T_eff,r_star,Lum)

    # -------------------------------------------------------------
    # modeling
    # -------------------------------------------------------------

    for spec_path,outputdir,distance,T_star,r_star,lum_star,wl_filter in zip(spec_path_lst,output_path_lst,distance_lst,T_star_lst,r_star_lst,lum_star_lst,wl_filter_lst):
        target_data_label = os.path.basename(spec_path).split('.')[0]
        print('-------------------------------------')
        print(run_name)
        print('    '+target_data_label)
        print('-------------------------------------')
        t1 = time.perf_counter()

        r_in = 0.069*np.sqrt(lum_star)
        # print('%.2f'%r_in)

        # -------------------------------------------------------------
        # read input spectrum and opacity files
        # -------------------------------------------------------------

        #read input spectrum
        # ==================

        print('Reading input data: '+str(spec_path))
        wl,fluxdata,fluxerr = np.loadtxt(spec_path, comments="#", skiprows=1, usecols=(0,1,2), unpack=True)

        nans,x = nan_helper(fluxdata)
        fluxdata = fluxdata[~nans]
        wl = wl[~nans]
        fluxerr = fluxerr[~nans]

        wl_idx = np.logical_and(wl>=setup.wl_limits[0],wl<=setup.wl_limits[1])
        wl = wl[wl_idx]
        fluxdata = fluxdata[wl_idx]
        fluxerr = fluxerr[wl_idx]
        #fluxerr_tmp = fluxerr_tmp[wl_idx]
        #fluxerr = 0.0*fluxdata+np.nanmedian(fluxerr_tmp)

        # apply wl filter
        wl_idx = np.logical_and(wl>100.0,wl<=0.0) #all False
        #print('initial',wl)
        for wl_range in wl_filter:
            tmp_idx = np.logical_and(wl>wl_range[0],wl<=wl_range[1])
            wl_idx = np.logical_or(wl_idx,tmp_idx)
        fluxdata = fluxdata[~wl_idx]
        fluxerr = fluxerr[~wl_idx]
        wl=wl[~wl_idx]
        N_data = len(fluxdata)
        freq = (c0*1e6)/wl
        #print('final',wl)
        freq_lst.append(freq)
        wl_lst.append(wl)
        fluxdata_lst.append(fluxdata)

        
        # Read opacity files
        # ==================
        
        print('Reading opacity files: ')
        kappa_arr = np.zeros((len(wl),len(opac_fname_list)))
        for i,fname in enumerate(opac_fname_list):
            print('    '+str(fname))
            if opac_type == 'Q_abs':
                N,gsize,gdens=np.loadtxt(opac_dir / fname,max_rows=1)
            n_comment_lines = 0
            with open(opac_dir / fname) as f:
                for line in f:
                    if line.startswith('#'):
                        n_comment_lines += 1
                    else:
                        break
            if opac_type == 'Q_abs':
                wl_Q,opac_data = np.loadtxt(opac_dir / fname, comments="#", skiprows=n_comment_lines+2, usecols=(0,1), unpack=True)
                kappa_arr[:,i] = np.interp(wl, wl_Q, 0.1* opac_data* 3.0 / (4.0 * gsize * 1e-4 * gdens)) #m^2/kg
            if opac_type == 'kappa_abs':
                # GRF opacity files: lambda[um], kappa_tot, kappa_abs, kappa_sca
                # optool opacity files: lambda[um]  kabs [cm^2/g]  ksca [cm^2/g]    g_asymmetry
                if n_comment_lines > 20:
                    usec = (0,1) # optool opacity file
                    # print('optool opacity file')
                else: 
                    usec = (0,2) #GRF opacity files
                    # print('GRF opacity file')
                wl_Q,opac_data = np.loadtxt(opac_dir / fname, comments="#", skiprows=n_comment_lines+2, usecols=usec, unpack=True)
                kappa_arr[:,i] = np.interp(wl, wl_Q, 0.1* opac_data) #convert to m^2/kg

        print('-------------------------------------')

        # -------------------------------------------------------------
        # Setup model parameters
        # -------------------------------------------------------------

        kappa_arr_lst.append(kappa_arr)
        param_dic = {}
        kappa_label_list = []
        kappa_label_short_list = []
        grain_name_list = []
        grain_size_list = []
        
        if setup.fit_mode == 'full':
            dust_coeff_free = True
        elif setup.fit_mode == 'with_nnls':
            dust_coeff_free = False


        if setup.fit_two_zones:
            param_dic['r_z'] = {'value':1.0,'limits':[r_in/5.0,10.0],'free':True}
            param_dic['n_z'] ={'value':2,'limits':[1,2],'free':False}
            zone_tags = ['_in','_out']
        else:
            param_dic['n_z'] ={'value':1,'limits':[1,2],'free':False}
            zone_tags = ['']

        for zone_tag in zone_tags:
            for Q_fname in opac_fname_list:
                label = os.path.splitext(Q_fname)[0]+zone_tag
                kappa_label_list.append(label)
                kappa_label_short_list.append((get_color_label(label))[1])
                # grain_name_list.append(re.sub(r'[0-9.]+', '', get_short_label(label)))
                # grain_size_list.append(re.sub("[^\d\.]", "", get_short_label(label)))
                grain_name_list.append((get_color_label(label))[1])
                grain_size_list.append((get_color_label(label))[2])
                param_dic[label]={'value':5e-4,'limits':[0.0,5e-2],'free':dust_coeff_free, #np.random.uniform(low=0.0, high=5e-2) # np.log10(2e-4) #'limits':[np.log10(1e-9),np.log10(5e-2)]
                                'grain_name':grain_name_list[-1] ,'grain_size':grain_size_list[-1],'mass_fraction':np.nan}
        param_dic['r_in'] =       {'value':r_in,'limits':[r_in/5.0,1.0],'free':True}
        param_dic['r_out'] =      {'value':20.0,'limits':[0.0,40.0],'free':False} #
        param_dic['T_dust_in'] =  {'value':1200,'limits':[400.0,1700.0],'free':True} ##
        param_dic['q_dust'] =     {'value':-0.7,'limits':[-3.0,0.0],'free':True} ##
        param_dic['T_midplane_in'] = {'value':800,'limits':[100.0,1500.0],'free':True}
        param_dic['q_midplane'] = {'value':-1.4,'limits':[-3.0,0.0],'free':True}
        param_dic['rim_width'] =  {'value':0.01,'limits':[0.0,2.0],'free':True}
        param_dic['T_rim_in'] =   {'value':1400.0,'limits':[800.0,1800.0],'free':True}
        param_dic['q_rim'] =      {'value':0.0,'limits':[-3.0,0.0],'free':False}
        param_dic['T_star'] =     {'value':T_star,'limits':[2000.0,20000.0],'free':False}
        param_dic['r_star'] =     {'value':r_star,'limits':[0.0,100.0],'free':False} #in Solar radii
        param_dic['n_r'] =        {'value':100,'limits':[0.0,1.0],'free':False}
        param_dic['n_rim'] =      {'value':10,'limits':[0.0,1.0],'free':False}
        param_dic['gap_width'] =  {'value':0.0,'limits':[0.0,4.0],'free':False}
        param_dic['r_gap'] =      {'value':0.0,'limits':[0.0,15.0],'free':False}
        param_dic['dscale'] =     {'value':distance*parsec/au,'limits':[-1.0,1.0],'free':False}
        #param_dic['AV'] =         {'value':0.0,'limits':[0.0,10.0],'free':False}

        if setup.fit_two_zones:
            param_dic['T_dust1'] =  {'value':900,'limits':[600.0,1700.0],'free':True} #
            param_dic['T_dust2'] =  {'value':400,'limits':[ 50.0,1000.0],'free':True} # 
            param_dic['r_out']['free'] = True
            param_dic['T_dust_in']['free'] = False
            param_dic['q_dust']['free'] = False
            
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

        # prev_coeffs = np.zeros(len(kappa_label_list))
        # param_dic['prev_coeffs'] = {'value':prev_coeffs,'limits':[0.0,1.0],'free':False}

        grain_name_list_unique,grain_size_counts = np.unique(grain_name_list,return_counts=True)

        # mass fractions
        n_free_params = 0
        free_param_labels = []
        free_param_values = []

        for label,param in param_dic.items():
            # print(label,param['free'])
            if param['free'] == True:
                n_free_params+=1
                free_param_labels.append(label)
                free_param_values.append(param['value'])

        # -------------------------------------------------------------
        # Plot initial model against data (optional)
        # -------------------------------------------------------------

        output_path = outputdir / (run_name+'_'+target_data_label +'_fit_init_'+setup.fit_method+'.png')
        model_flux = model_fn(freq,param_dic,[kappa_arr,kappa_label_list],setup.fit_mode,fluxdata)
        flux_dust,flux_midplane,flux_star,flux_rim = get_component_fluxes(freq,param_dic,[kappa_arr,kappa_label_list],setup.fit_mode,fluxdata)
        model_fluxes = [model_flux,np.nansum(flux_dust,axis=1),flux_midplane,flux_star,flux_rim,flux_dust]
        chi2_red = -2.0*lnlike(free_param_values, freq, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],setup.fit_mode])
        if setup.do_plot_init:
            plot_title = target_data_label.replace('_MINDS_MIRI','')+r' init, $\chi^2_\mathrm{red}$ = %.2f'%(chi2_red)
            plot_fit(output_path,wl,fluxdata,fluxerr,model_fluxes,kappa_label_list,plot_title,param_dic,setup.fit_mode,xlim=setup.wl_limits)

            print('Mass fractions (%)')
            surfdens = 0.0
            if setup.fit_mode == 'with_nnls':
                for kappa_label in kappa_label_list:
                    #surfdens += 10.0**param_dic[kappa_label]['value']
                    surfdens += param_dic[kappa_label]['value']
                for kappa_label in kappa_label_list:
                    # mf = 10.0**param_dic[kappa_label]['value']/surfdens*100.0
                    mf = param_dic[kappa_label]['value']/surfdens*100.0
                    print('%-38s %.3f'%(kappa_label,mf))

        if setup.do_fit:
            # -------------------------------------------------------------
            # Setup fitter
            # -------------------------------------------------------------

            # if setup.fit_method == 'mcmc':
            #     nwalkers = n_free_params * 2
            #     burnin = int(n_steps/2)

            #     # initial positions of the walkers
            #     pos = [] 
            #     for param in param_dic.values():
            #         if param['free'] == True:
            #             margin=0.001*(param['limits'][1]-param['limits'][0])
            #             pos.append(np.random.uniform(low=param['limits'][0]+margin,
            #                 high=param['limits'][1]-margin,size=nwalkers))
            #     pos = np.transpose(np.array(pos))

            #     # run MCMC
            #     with mp.get_context('fork').Pool(int(mp.cpu_count()/2)) as pool:
            #         sampler = emcee.EnsembleSampler(nwalkers, n_free_params, lnprob, 
            #                 args=[freq, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list]]],pool=pool)
            #         sampler.run_mcmc(pos, n_steps, progress=True)
                    
            # elif setup.fit_method == 'dynesty':
            priors = np.array([(param_dic[key]['limits'][0], param_dic[key]['limits'][1]) for key in free_param_labels])
            # Nested sampling. #int(mp.cpu_count()/2)
            if __name__ == "__main__":
                with dynesty.pool.Pool(2, lnlike, prior_transform, 
                        logl_args=[freq, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],setup.fit_mode]],
                        ptform_args=[priors]) as pool:
                    sampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, 
                        n_free_params, pool = pool,bound='multi',
                        sample='rwalk') 

                    # -------------------------------------------------------------
                    # Do fit
                    # -------------------------------------------------------------
                    print('Do fit')
                    sampler.run_nested(dlogz_init=0.05, nlive_init=500, nlive_batch=100,
                        maxiter=setup.maxiter,
                        checkpoint_file=str(outputdir / (run_name+'_'+target_data_label +'_dynesty.save')),
                        print_progress=True)
                                    # maxiter=2000,use_stop=False)
                        # dynesty.DynamicNestedSampler(... , bound='single')
                        # sample='slice')  #slow
                    # sampler =  dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                    # n_free_params, pool = pool,nlive=n_live)
                    #sampler = dynesty.NestedSampler(lnlike, prior_transform, n_free_params,
                    #    logl_args=[freq, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],fit_mode]],
                    #    ptform_args=[param_dic,free_param_labels],nlive=n_live)

                # -------------------------------------------------------------
                # Save and plot fit results
                # -------------------------------------------------------------
                print('-------------------------------------')
                print('Save and plot fit results')

                # Determine best-fit parameter values
                # ===================================

                # if setup.fit_method == 'mcmc':
                #     chi2_red = -2.0*sampler.get_log_prob(discard = burnin, flat=True)
                #     samples = sampler.get_chain(discard = burnin, flat = True)
                #     confidence_intervals = list(map(lambda v: (v[0], v[1]),
                #         zip(*np.percentile(samples, [16, 84],axis=0))))
                #     idx = np.argmin(chi2_red)
                #     chi2r_fit = chi2_red[idx]
                #     print('Min chi2:',chi2r_fit)
                #     best_parameters = samples[idx]
                # elif setup.fit_method == 'dynesty':
                dy_result = sampler.results
                chi2_red = -2.0*dy_result.logl
                samples = dy_result.samples
                weights = dy_result.importance_weights()
                confidence_intervals = [dyfunc.quantile(samps, [0.16, 0.84], weights=weights)
                    for samps in samples.T]

                # Compute weighted mean and covariance.
                # mean, cov = dyfunc.mean_and_cov(samples, weights)
                # best_parameters = mean
                idx = np.argmin(chi2_red)
                chi2r_fit = chi2_red[idx]
                # print('Min chi2:',chi2r_fit)
                best_parameters = samples[idx]
            
                for (pvalue,label,confidence_interval) in zip(best_parameters,free_param_labels, confidence_intervals):
                    param_dic[label]['value'] = pvalue
                    param_dic[label]['err_l'] = pvalue-confidence_interval[0]
                    param_dic[label]['err_u'] = confidence_interval[1]-pvalue
                    param_dic[label]['conf_l'] = confidence_interval[0]
                    param_dic[label]['conf_u'] = confidence_interval[1]
                
                #     print(label,pvalue)
                # if setup.fit_method == 'dynesty':
                    # chi2r_fit = -2.0*lnlike(best_parameters, wl, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],lnlike])

                model_flux = model_fn(freq,param_dic,[kappa_arr,kappa_label_list],setup.fit_mode,fluxdata)
                flux_dust,flux_midplane,flux_star,flux_rim = get_component_fluxes(freq,param_dic,[kappa_arr,kappa_label_list],setup.fit_mode,fluxdata)

                # Save best-fit parameters 
                # ========================

                outfile_path = outputdir / (run_name+'_'+target_data_label +'_fit_best_params.txt')
                with open(outfile_path, "w") as f:
                    f.write("Chi2r : {}\n".format(chi2r_fit))
                    f.write('Fit method : %s\n'%(setup.fit_method))
                    # if setup.fit_method == 'mcmc':
                    #     f.write('nsteps : %d\n'%(n_steps))
                    #     f.write('ndiscard : %d\n'%(burnin))
                    #     f.write('nwalkers : %d\n'%(nwalkers))
                    # elif setup.fit_method == 'dynesty':
                    #     f.write('nlive : %d\n'%(n_live))
                    f.write('niter : %d\n'%(dy_result['niter']))
                    f.write("----------------\n")
                    f.write("Dust mass fractions (%)\n")
                    
                    print("Best-fit dust mass fractions (%)")
                    surfdens = 0.0
                    for kappa_label in kappa_label_list:
                        if setup.fit_mode == 'full':
                            surfdens += 10.0**param_dic[kappa_label]['value']
                        elif setup.fit_mode == 'with_nnls':
                            surfdens += param_dic[kappa_label]['value']
                    for kappa_label in kappa_label_list:
                        if setup.fit_mode == 'full':
                            val = 10.0**param_dic[kappa_label]['value']/surfdens*100.0
                            if param_dic[kappa_label]['free'] == True:
                                err_l = (10.0**param_dic[kappa_label]['value']-10.0**param_dic[kappa_label]['conf_l'])/surfdens*100.0
                                err_u = (10.0**param_dic[kappa_label]['conf_u']-10.0**param_dic[kappa_label]['value'])/surfdens*100.0
                                outstr = '%-40s %7.3f (-%7.3f/+%7.3f)'%(kappa_label,val,err_l,err_u)
                            else: 
                                outstr = '%-40s %7.3f'%(kappa_label,val)
                        elif setup.fit_mode == 'with_nnls':
                            val = param_dic[kappa_label]['value']/surfdens*100.0
                            #err_l = (param_dic[kappa_label]['value']-param_dic[kappa_label]['conf_l'])/surfdens*100.0
                            #err_u = (param_dic[kappa_label]['conf_u']-param_dic[kappa_label]['value'])/surfdens*100.0
                            outstr = '%-40s %7.3f (-%7.3f/+%7.3f)'%(kappa_label,val,0,0) #err_l,err_u)
                        print('%-40s %7.3f'%(kappa_label,val))
                        f.write(outstr+'\n')
                    f.write("----------------\n")
                    print('-------------------------------------')
                    f.write("Free parameters\n")
                    print("Free parameters")
                    i = 0
                    for (pvalue,label,confidence_interval) in zip(best_parameters,free_param_labels,confidence_intervals):
                        outstr = '%s: %.10f (-%.10f/+%.10f)'%(label,pvalue,pvalue-confidence_interval[0],confidence_interval[1]-pvalue)
                        f.write(outstr+'\n')
                        print(outstr)
                        i+=1
                    f.write("----------------\n")
                    f.write("Fixed parameters\n")
                    i = 0
                    for key, param in param_dic.items():
                        if param['free'] == False:
                            try:
                                f.write('%s: %.15f\n'%(key,param['value']))
                            except TypeError as e:
                                pass
                            i+=1
                    f.write("----------------\n")
                    f.write("Input spectrum\n")
                    f.write('%s\n'%(spec_path))
                    f.write("----------------\n")
                    f.write("Input opacity files\n")
                    for fname in opac_fname_list:
                        f.write('%s\n'%(fname))
                    print('-------------------------------------')

                # Plot best-fit model against data 
                # ================================     
                N_fr = 0
                #N_data
                for key in param_dic:
                    if param_dic[key]['free'] == True:
                        N_fr+=1
                N_free = N_fr + param_dic['n_z']['value']*len(kappa_label_list)
                chi2r_recalc = chi2r_fit*N_data/(N_data-N_free)
                print('N_model = %d, N_dustcomp = %d, N_free = %d, N_data = %d'%(N_fr,param_dic['n_z']['value']*len(kappa_label_list),N_free,N_data))
                print("chi2r_fit = %.2f, chi2r_recalc = %.2f"%(chi2r_fit,chi2r_recalc))
                if 'AATau' in target_data_label or 'FTTau' in target_data_label or\
                    'LkCa15' in target_data_label or 'Sz98' in target_data_label:
                    leg_ncol = 1
                    leg_loc = 'right'
                else:
                    leg_ncol = 3
                    leg_loc = 'best'

                if 'spectrum_rb' in target_data_label:
                    plot_title = target_data_label.replace('postprocess_spectrum_rb_epoch','Epoch') +', $\chi^2_\mathrm{r}$ = %.2f'%(chi2r_recalc)
                else:
                    plot_title = target_data_label.replace('_MINDS_MIRI','').replace('_smoothed_rebinned','') +', $\chi^2_\mathrm{r}$ = %.2f'%(chi2r_recalc)

                model_fluxes = [model_flux,np.nansum(flux_dust,axis=1),flux_midplane,flux_star,flux_rim,flux_dust]
                #plot_title = target_data_label.replace('_MINDS_MIRI','').replace('_smoothed_rebinned','')+r' best-fit, $\chi^2_\mathrm{red}$ = %.2f'%(chi2r_fit)
                output_path = outputdir / (run_name+'_'+target_data_label +'_fit_'+setup.fit_method+'.png')
                plot_fit(output_path,wl,fluxdata,fluxerr,model_fluxes,kappa_label_list,plot_title,param_dic,setup.fit_mode,
                        xlim=[setup.wl_limits[0],setup.wl_limits[1]],wl_filter=wl_filter,leg_loc=leg_loc,leg_ncol=leg_ncol)
                #plot_title = target_data_label.replace('_MINDS_MIRI','').replace('_smoothed_rebinned','')
                output_path = outputdir / (run_name+'_'+target_data_label +'_fit_'+setup.fit_method+'.pdf')
                plot_fit(output_path,wl,fluxdata,fluxerr,model_fluxes,kappa_label_list,plot_title,param_dic,setup.fit_mode,
                        xlim=[setup.wl_limits[0],setup.wl_limits[1]],wl_filter=wl_filter,leg_loc=leg_loc,leg_ncol=leg_ncol)
                # save data and model spectra
                outfile_path = outputdir / (run_name+'_'+target_data_label +'_fit_model_spectra.txt')
                with open(outfile_path, "w") as f:
                    f.write('# Wl(um) F_data(Jy) err_F_data(Jy)  F_tot(Jy) F_dust(Jy) F_midp(Jy) F_star(Jy)  F_rim(Jy) ')
                    for kappa_label in kappa_label_list:
                        f.write('F_nu_'+(get_color_label(kappa_label))[1]+'(Jy) ')
                    f.write('\n')
                    for i in range(len(wl)):
                        f.write('%8.5f %.6e %.6e %.6e %.6e %.6e %.6e %.6e '%(wl[i], fluxdata[i], fluxerr[i],  model_flux[i],(np.nansum(flux_dust,axis=1))[i],flux_midplane[i],flux_star[i],flux_rim[i]))
                        for j in range(len(flux_dust[0,:])):
                            f.write('%.6e '%flux_dust[i,j])
                        f.write('\n')

                # corner plot
                # ===========
                short_param_labels = []
                for label in free_param_labels:
                        #print(label,get_color_label(label) )
                        short_param_labels.append((get_color_label(label))[1])
                try:
                    output_path = outputdir / (run_name+'_'+target_data_label +'_'+setup.fit_method+'_corner_plot.png')
                    #print(short_param_labels)
                    # if setup.fit_method == 'mcmc':
                    #     fig = corner.corner(samples, labels=short_param_labels,quantiles=[0.16,0.5,0.84],
                    #                         show_titles=False,use_math_text=True,
                    #                         truths=best_parameters, bins=40,label_kwargs={'fontsize':10},
                    #                         range=[0.999]*n_free_params) #bins=50

                    # elif setup.fit_method == 'dynesty':
                    fig, ax = dyplot.cornerplot(dy_result, color='blue', truths=best_parameters,
                                            truth_color='black', show_titles=True,labels=short_param_labels,
                                            max_n_ticks=3, quantiles=[0.16,0.50,0.84],
                                            use_math_text=True) #title_fmt='.2e'
                                            #fig=(fig, axes[:, :3]))
                    fig.savefig(output_path,dpi=200)
                    #plt.show()
                    plt.close(fig)
                except ValueError as e:
                    print('Cannot make corner plot.')

                # run plot
                # ===========
                try:
                    output_path = outputdir / (run_name+'_'+target_data_label +'_'+setup.fit_method+'_run_plot.png')
                    lnz_truth = N_free * -np.log(2 * 10.)  # analytic evidence solution
                    fig, axes = dyplot.runplot(dy_result, lnz_truth=lnz_truth,use_math_text=True)  # summary (run) plot
                    fig.savefig(output_path,dpi=200)
                    #plt.show()
                    plt.close(fig)
                except ValueError as e:
                    print('Cannot make run plot.')

                # trace plot
                # ===========
                try:
                    output_path = outputdir / (run_name+'_'+target_data_label +'_'+setup.fit_method+'_trace_plot.png')
                    fig, ax = dyplot.traceplot(dy_result,show_titles=True,labels=short_param_labels,
                            truths=best_parameters,truth_color='black',use_math_text=True,
                            trace_cmap='viridis',quantiles=[0.16,0.50,0.84])
                    fig.savefig(output_path,dpi=200)
                    #plt.show()
                    plt.close(fig)
                except ValueError as e:
                    print('Cannot make trace plot.')

                output_path = outputdir / (run_name+'_'+target_data_label +'_'+setup.fit_method+'_chi2_plot.png')
                best = np.minimum.accumulate(chi2_red)
                fig, ((ax)) = plt.subplots(1, 1, sharey=False, sharex=False,figsize=(8,6))
                ax.step(np.arange(len(best)),
                         best,where='post')
                ax.grid(which='minor',alpha=0.3)
                ax.grid(which ='major')
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Best $\chi^2$")
                ax.set_yscale('log')
                fig.savefig(output_path,dpi=200) 
                plt.close(fig)

            elapsed_time = time.perf_counter() - t1
            print('Finished '+target_data_label+' in %.1f min'%(elapsed_time/60.0))
            print('-------------------------------------')
        elapsed_time = time.perf_counter() - t0
        print('Finished '+run_name+' in %.1f min'%(elapsed_time/60.0))
        print('-------------------------------------')


# %%

################################################################
# Postprocessing: read previous results, make plots, 
# and calculate uncertainties of the mass fractions
################################################################

if setup.load_previous_results:
    from tqdm import tqdm

    run_name = '36_newDHS_Qtemp_300K_with_ann_SiO2_nobigcrystals_3gs'

    for spec_path,result_dir,distance,T_star,r_star,lum_star,wl,freq,fluxdata,kappa_arr in zip(spec_path_lst,output_path_lst, distance_lst,T_star_lst,r_star_lst,lum_star_lst,wl_lst,freq_lst,fluxdata_lst,kappa_arr_lst):
        target_data_label = os.path.basename(spec_path).split('.')[0]
        print(target_data_label)
        outputdir = result_dir

        model_path = result_dir / run_name+'_'+target_data_label +'_fit_model_spectra.txt'
        model_data = np.loadtxt(model_path) #,usecols=(0,1,2,3,4,5,6,7),unpack=True)
        
        f = open(model_path)
        header = f.readline()
        f.close()

        #wl = model_data[:,0]
        #fluxdata = model_data[:,1]
        model_flux = model_data[:,3]
        flux_dust_sum = model_data[:,4]
        flux_midplane = model_data[:,5]
        flux_star = model_data[:,6]
        flux_rim = model_data[:,7]
        flux_dust = model_data[:,8:]
       
        #read best-fit parameters 
        param_dic = {}
        kappa_label_list = []
        best_params_file_path = result_dir / run_name+'_'+target_data_label +'_fit_best_params.txt'
        free_param_start = False
        fix_param_start = False
        dust_param_start = False

        with open(best_params_file_path, "r") as f:
            for line in f:
                if line.startswith('--------'):
                    free_param_start = False
                    fix_param_start = False
                    dust_param_start = False
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
                    param_dic[label] = {'value':float((line.split())[1]),'limits':[np.nan,np.nan],'free':False,
                                                    'grain_size':gs,'grain_name':gn}
                if dust_param_start:
                    kappa_label_list.append((line.split())[0].replace(':',''))
                if line.startswith('Free parameters'):
                    free_param_start = True
                if line.startswith('Fixed parameters'):
                    fix_param_start = True
                if line.startswith('Dust mass fractions'):
                    dust_param_start = True

        if setup.plot_previous_results == True:
            #read best-fit parameters 
            param_dic,kappa_label_list,opac_fname_list,chi2r = read_fit_param_file(best_params_file_path)     
            N_fr = 0
            N_data = len(fluxdata)
            for key in param_dic:
                if param_dic[key]['free'] == True:
                    N_fr+=1
            N_free = N_fr + param_dic['n_z']['value']*len(kappa_label_list)
            chi2r_recalc = chi2r*N_data/(N_data-N_free)

            # plot best-fit model against data
            output_path = outputdir / 'extra_plots' / (run_name+'_'+target_data_label +'_fit_'+setup.fit_method+'.pdf')
            #print(output_path)
            model_fluxes = [model_flux,flux_dust_sum,flux_midplane,flux_star,flux_rim,flux_dust]
            #plot_title = target_data_label.replace('_MINDS_MIRI','') #+' best-fit, $\chi^2_\mathrm{red}$ = %.2f'%(chi2r_fit)
            plot_title = target_data_label.replace('_MINDS_MIRI','').replace('_smoothed_rebinned','') +', $\chi^2_\mathrm{r}$ = %.2f'%(chi2r_recalc)

            plot_fit(output_path,wl,fluxdata,fluxerr,model_fluxes,kappa_label_list,
                plot_title,param_dic,setup.fit_mode,xlim=[setup.wl_limits[0],setup.wl_limits[1]],
                wl_filter=wl_filter,axr_ylim=[-6.0,6.0]) #leg_loc=leg_loc,leg_ncol=leg_ncol
            
        if setup.calc_errors:
            print('Compute uncertainties')
            save_modes = ['eq','raw'] #'raw' 'eq'
            for save_mode in save_modes:
                print('Saving samples: '+save_mode)
                savefile = result_dir / run_name+'_'+target_data_label +'_dynesty.save'
                with open(savefile, "rb") as fs:
                    res = pickle.load(fs)
                    sampler = res['sampler']
                    results = sampler.results
                    samples_raw = results.samples
                    weights = results.importance_weights()
                    #confidence_intervals = [dyfunc.quantile(samps, [0.16, 0.84], weights=weights) for samps in samples.T]
                    samples_equal = dyfunc.resample_equal(samples_raw, weights)
                    #p16, p84 = np.percentile(samples_equal[:, 0], [16, 84])
                    n_samples = len(samples_equal[:,0])
                    mass_fraction_samples = np.zeros((n_samples,len(kappa_label_list) ) )
                    intflux_samples = np.zeros((n_samples,len(kappa_label_list) ) )
                    logl = results.logl     # log-likelihood values
                    chi2 = -2 * logl
                    #print(chi2)    
                    #chi2_min = np.nanmin(chi2)
                    #i=-1
                    #print(wl)
                    # 1 Jy = 1e-26 W m-2 Hz-1   
                                 
                    outfile_path = outputdir / (run_name+'_'+target_data_label +'_geom_'+save_mode+'samples.dat')
                    f = open(outfile_path, "w") # save (equally weighted or raw) geometric parameter samples
                    f.write('# ')
                    for free_param_label in free_param_labels:
                        f.write('%s '%(free_param_label))
                    f.write('\n')
                    
                    for i in tqdm(range(n_samples)):
                        if save_mode == 'raw':
                            free_param_values = samples_raw[i,:]
                        elif save_mode == 'eq':
                            free_param_values = samples_equal[i,:]
                        for (pvalue,label) in zip(free_param_values,free_param_labels):
                            #print(label,pvalue)
                            param_dic[label]['value'] = pvalue
                        
                        #model_flux = model_fn(freq,param_dic,[kappa_arr,kappa_label_list],setup.fit_mode,fluxdata)
                        #print(param_dic)
                        flux_dust,flux_midplane,flux_star,flux_rim = \
                            get_component_fluxes(freq,param_dic,[kappa_arr,kappa_label_list],\
                            setup.fit_mode,fluxdata)
                        #print(flux_dust)

                        surfdens = 0.0
                        for kappa_label in kappa_label_list:
                            surfdens += param_dic[kappa_label]['value']
                            #print(kappa_label,param_dic[kappa_label]['value'])
                            
                        for j,kappa_label in enumerate(kappa_label_list):
                            val = param_dic[kappa_label]['value']/surfdens*100.0
                            mass_fraction_samples[i,j] = val
                            intflux_samples[i,j] = np.trapz(1e-26*np.flip(flux_dust[:,j]),x=np.flip(freq))
                        
                        for j in range(len(free_param_labels)):
                            f.write('%11.6f '%(free_param_values[j]))
                        f.write('%.3f\n'%chi2[i])
                                #print(kappa_label,val)
                        #print(mass_fraction_arr)
                    
                    f.close()
                    print('%-40s %5s %5s %5s'%('Dust species','p16','p50','p84'))
                    for i,kappa_label in enumerate(kappa_label_list):
                        p16,p50,p84 = np.nanpercentile(mass_fraction_samples[:, i], [16, 50, 84])
                        print('%-40s %5.2f %5.2f %5.2f'%(kappa_label,p16,p50,p84))
                    
                # save fraction samples
                outfile_path = outputdir / (run_name+'_'+target_data_label +'_mf_'+save_mode+'samples.dat')
                with open(outfile_path, "w") as f:
                    f.write('# ')
                    for kappa_label in kappa_label_list:
                        f.write('%s '%(kappa_label))
                    f.write('\n')
                    for i in range(n_samples):
                        for j in range(len(kappa_label_list)):
                            f.write('%.7f '%(mass_fraction_samples[i,j]))
                        f.write('%.6f\n'%chi2[i])

                # save intfluxsamples
                outfile_path = outputdir / (run_name+'_'+target_data_label +'_intflux_'+save_mode+'samples.dat')
                with open(outfile_path, "w") as f:
                    f.write('# ')
                    for kappa_label in kappa_label_list:
                        f.write('%s '%(kappa_label))
                    f.write('\n')
                    for i in range(n_samples):
                        for j in range(len(kappa_label_list)):
                            f.write('%6e '%(intflux_samples[i,j]))
                        f.write('%.6f\n'%chi2[i])

print('EXTERMINATE')
