import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as apyco
import astropy.constants as apyconst
from scipy.interpolate import interp1d
import pandas as pd
from scipy import stats
import legwork as lw
import paths
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import healpy as hp
import os


def sample_porb_from_Pala_2020(nCV, rng):
    lpdist = pd.read_csv(paths.data / "lpdist3.out", delim_whitespace=True, header=None, names=['logp', 'CDF'])


    lpdist['porb'] = 10**lpdist['logp'] # convert logp to porb
    lpdist['porb'] = lpdist['porb'] / 60 # convert minutes to hours

    # Smooth the CDF with a kde to make it more continuous
    #lpdist['CDF'] = scipy.ndimage.gaussian_filter1d(lpdist['CDF'], 2)

    # sample from the CDF
    pp = rng.uniform(0, 1, nCV)
    porb = np.interp(pp, lpdist.CDF, lpdist.porb)

    return porb


def sample_position_from_Pala_2020(rng, rho_0=4.8e-6, h=280, dist_max=600):
    # LSS dist_max is given in pc
    # x, y, z returned are in kpc

    # the Pala+2020 model is just a cylinder with an exponential decay in z
    # this means we can assign x, y randomly in a circle and z with the exponential decay
    N_sample_positive = rho_0 * np.pi *dist_max**2 * h * (1 - np.exp(-((dist_max)/h)))
    N_sample_total = 2 * N_sample_positive

    # we will do a rejection sample to get the correct number of sources.
    # we will sample 5 times the number we need and then downsample
    extraFactor = 5

    # determine if we add the remainder of the decimal as a source
    prob_extra = rng.uniform(0, 1)
    remainder = N_sample_total - int(N_sample_total)
    if prob_extra < remainder:
        N_sample_total = int(N_sample_total) + 1
    else:
        N_sample_total = int(N_sample_total)
    
    # uniform in a disk around the sun 
    # kb is lazy and will do a rejection sample
    x = rng.uniform(-dist_max, dist_max, extraFactor*N_sample_total)
    y = rng.uniform(-dist_max, dist_max, extraFactor*N_sample_total)
    r = np.sqrt(x**2 + y**2)
    # first take everything within the disk
    ind_keep, = np.where(r < dist_max)
    x = x[ind_keep]
    y = y[ind_keep]
    
    # next downsample to size of population
    x = x[:N_sample_total]
    y = y[:N_sample_total]
    
    # next assign the z population
    z = rng.exponential(scale=h, size=5*N_sample_total)
    plane_sample = rng.uniform(0, 1, 5*N_sample_total)
    z[plane_sample < 0.5] = -z[plane_sample < 0.5]
    # filter to systems within dist_max
    z = z[abs(z) < dist_max]
    z = z[:N_sample_total]
    
    ## now place the final volume limit
    ind_volume_limit, = np.where(np.sqrt(x**2 + y**2 + z**2) < dist_max)
    x = x[ind_volume_limit]
    y = y[ind_volume_limit]
    z = z[ind_volume_limit]

    return x/1000, y/1000, z/1000 # LSS convert to kpc
    

def calculate_m2_from_porb(porb):
    # calculate the m2 mass from the orbital period by interpolating 
    # to find the time at P and m2 at that time
    dat = pd.read_csv(paths.data / 'kniggeTable.csv')
    t_interp = interp1d(dat['Per'], dat['logt'], fill_value = 'extrapolate')
    t_bin = t_interp(porb)
    m2_interp = interp1d(dat['logt'], dat['M2'], fill_value = 'extrapolate')
    m2 = m2_interp(t_bin)
    m2[porb < min(dat["Per"])] = min(m2)
    return m2
    

def get_Pala_sample(mu_m1, sigma_m1, sigma_m2, rng):
    pala2020 = pd.read_hdf(paths.data / 'Pala_2020_dat_combo.h5', key='dat')
    c = SkyCoord(pala2020.ra.values * u.deg, pala2020.dec.values * u.deg, distance=pala2020.distance.values * u.pc)
    c = c.transform_to(frame='galactic')
    x = c.cartesian.x.value
    y = c.cartesian.y.value
    z = c.cartesian.z.value
    porb = pala2020['porb'].values / 60 #convert mins to hours
    m2 = calculate_m2_from_porb(porb)
    m2_err = rng.normal(loc=0, scale=sigma_m2, size=len(porb))
    m2 = m2 + m2_err
    m1 = rng.normal(loc=mu_m1, scale=sigma_m1, size=len(porb))
    inclination = np.arccos(rng.uniform(-1, 1, len(porb)))
    return m1, m2, porb, x/1000, y/1000, z/1000, inclination

def sample_kpc_population(max_distance, mu_m1, sigma_m1, sigma_m2, rng):
    

    # first sample the population
    # sample the population positions and size based on Pala+2020 distribution & space density
    x, y, z = sample_position_from_Pala_2020(rho_0=4.8e-6, h=280, dist_max=max_distance)
    
    d = np.sqrt(x**2 + y**2 + z**2) * u.kpc
    ind_check, = np.where(d<0.15*u.kpc)
    while len(ind_check) < 42:
        print("We need at least 42 sources within 150pc. Generating new population!")
        x, y, z = sample_position_from_Pala_2020(rho_0=4.8e-6, h=280, dist_max=max_distance)
        d = np.sqrt(x**2 + y**2 + z**2) * u.kpc
        ind_check, = np.where(d<0.15*u.kpc)

    # assign a random inclination
    inclination = np.arccos(rng.uniform(-1, 1, len(x)))
    
    # sample the primary mass with normal distribution supplied by user
    m1 = rng.normal(loc=mu_m1, scale=sigma_m1, size=len(x))
    
    # get the orbital periods by sampling from the Pala+2020 table
    porb = sample_porb_from_Pala_2020(nCV=len(x), rng=rng)
    f_gw = 2/(porb * 3600) # this is simple because the binaries are circular and porb is in hrs

    # get the matching donor mass from the Knigge+2011 table
    m2 = calculate_m2_from_porb(porb)
    m2_err = rng.normal(loc=0, scale=sigma_m2, size=len(x))
    m2 = m2 + m2_err
    Pala_reassign = np.zeros(len(x))
    dat = np.vstack([m1, m2, f_gw, inclination, x, y, z, Pala_reassign]).T

    # next reassign some of the sources to match the Pala data exactly
    m1_P, m2_P, porb_P, x_P, y_P, z_P, inc_P = get_Pala_sample(mu_m1, sigma_m1, sigma_m2, rng)
    
    d = np.sqrt(dat[:,4]**2 + dat[:,5]**2 + dat[:,6]**2) * u.kpc
    ind_150, = np.where(d<0.15*u.kpc)

    # Some haking required here. Pala sample is 42 sources, so we need to randomly select 42 sources
    # from the 150pc sample and replace with the Pala sample.
    # But we also need to make sure that we don't replace the same source twice.
    ind_Pala = rng.random.choice(ind_150, len(m2_P), replace=False)   
    dat[ind_150, 7] = 2*np.ones(len(ind_150))

    dat[ind_Pala, 0] = m1_P
    dat[ind_Pala, 1] = m2_P
    dat[ind_Pala, 2] = 2/(porb_P*3600)
    dat[ind_Pala, 3] = inc_P
    dat[ind_Pala, 4] = x_P
    dat[ind_Pala, 5] = y_P
    dat[ind_Pala, 6] = z_P
    dat[ind_Pala, 7] = np.ones(len(m1_P))
    
    ind, = np.where(dat[:,7] > 0)

    c = SkyCoord(dat[:, 4], dat[:, 5], dat[:, 6], unit=u.kpc, frame='galactic', representation_type='cartesian')
    
    c_gal = c.transform_to('galactocentric')
    
    dat[:, 4] = c_gal.x
    dat[:, 5] = c_gal.y
    dat[:, 6] = c_gal.z

    return dat

def galactic_positions(size, rng, model="McMillan",disk='thick'):                        
    """Sample a set of Galactic positions of size=size distributed
    according to the user specified model. X,Y,Z positions in [pc];
    Galactocentric distance in [kpc]

    This has been adapted to match the normalization assumptions made in the BLIP model.

    Parameters
    ----------
    size : int
        Size of the sample
    model : str
        Current default model is 'McMillan'

    Returns
    -------
    xGx, yGx, zGx, inc, OMEGA, omega : array
        Array of sampled positions in Galactic cartesian coordinates
        centered on the Galactic center and orientations in radians
    
    """

    ## thick/thin switch
    if disk=='thick':
        zh = 0.9
    elif disk=='thin':
        zh = 0.3
    else:
        raise ValueError("disk must be 'thick' or 'thin'.")
    
    if model == "McMillan":
        r_save = []
        z_save = []
        # sample double exp func and then rejection sample
        while len(r_save) < size:
            rcut = 2.1
            q = 0.5
            r0 = 0.075
            alpha = -1.8
            rprime = np.sqrt(r**2 + (z/q)**2)
            ## ensures proper normalization for the rejection sampling
            rho_c = 0.45
            rh = 2.9
            r = rng.uniform(0.2, 20, size * 10)
            z = rng.uniform(0, 5, size * 10)
            prob = rng.uniform(0, 1, size * 10)
            bulge_sample_func = rho_c*np.exp(-(rprime ** 2 ) / rcut ** 2)
            # bulge_sample_func = 0
            disk_sample_func = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh) 
            actual_func = bulge_sample_func*(1 + rprime / r0)**(alpha) + disk_sample_func
            (indSave,) = np.where(prob < actual_func)
            for ii in indSave:
                r_save.append(r[ii])
                z_save.append(z[ii])
        r = np.array(r_save[:size])
        z = np.array(z_save[:size])
        ind_pos_neg = rng.uniform(0, 1, len(z))
        (ind_negative,) = np.where(ind_pos_neg > 0.5)
        z[ind_negative] = -z[ind_negative]

        # Assign the azimuthal positions:
        phi = rng.uniform(0, 2 * np.pi, size)

        # convert to cartesian:
        xGX = r * np.cos(phi)
        yGX = r * np.sin(phi)
        zGX = z
    elif model == "McMillan_fixed":
        ## this model samplesfrom the McMillan distribution but draws in [x,y,z] cartesian space to avoid singularity at r=0.
        x_save = []
        y_save = []
        z_save = []

        # sample double exp func and then rejection sample
        while len(z_save) < size:
            rcut = 2.1
            q = 0.5
            r0 = 0.075
            alpha = -1.8
            ## ensures proper normalization for the rejection sampling
            rho_c = 1
            rh = 2.9
            x = rng.uniform(-20,20, size * 10)
            y = rng.uniform(-20,20, size * 10)
            z = rng.uniform(0, 5, size * 10)
            r = np.sqrt(x**2 + y**2)
            prob = rng.uniform(0, 1, size * 10)
            
            rprime = np.sqrt(r**2 + (z/q)**2)
            bulge_sample_func = rho_c*np.exp(-(rprime/rcut)** 2)
            # bulge_sample_func = 0
            disk_sample_func = rho_c*np.exp(-r/rh)*np.exp(-np.abs(z)/zh) 
            actual_func = bulge_sample_func*(1 + rprime / r0)**(alpha) + disk_sample_func
            (indSave,) = np.where(prob < actual_func)
            for ii in indSave:
                x_save.append(x[ii])
                y_save.append(y[ii])
                z_save.append(z[ii])
        x = np.array(x_save[:size])
        y = np.array(y_save[:size])
        z = np.array(z_save[:size])
        ind_pos_neg = rng.uniform(0, 1, len(z))
        (ind_negative,) = np.where(ind_pos_neg > 0.5)
        z[ind_negative] = -z[ind_negative]

        xGX = x
        yGX = y
        zGX = z

    # assign an inclination, argument of periapsis, and longitude of ascending node
    inc = np.pi - np.arccos(rng.uniform(-1, 0, size))

    return xGX, yGX, zGX, inc


def sample_fullgxy_population(mu_m1, sigma_m1, sigma_m2, rng, disk='thin'):    
    """sample the MW's CV population, normalized by the Scaringi 2023 1kpc population based on Pala2020 space density measurements.

    Parameters
    ----------
    mu_m1 : float
        mean of the normal distribution to sample the primary mass from
    sigma_m1 : float
        standard deviation of the normal distribution to sample the primary mass from
    sigma_m2 : float
        standard deviation of the normal distribution to add to the donor mass calculated from the Knigge+2011 table
    rng : np.random.Generator
        random number generator

    Returns
    -------
    dat : array
        array of sampled CV population with columns: 
        m1[Msun], m2[Msun], f_gw[Hz], inclination[rad], x_galcent[kpc], y_galcent[kpc], z_galcent[kpc], Pala_reassigned (1 if reassigned to match Pala, 2 if in 150pc sample but not reassigned, 0 otherwise)
    """
    
    # To do, integrate EM gap fill within this function
    
    # LSS draw 5e6 galactic positions in galactocentric cartesian coord (kpc) and inclinations.
    overdraw = int(5e6)
    # we then normalize by the 1kpc population created in Scaringi 2023.
    #scaringi1kpc_pop = pd.read_csv(paths.data / "dat_maxDistance_1000_final.txt")
    #scar_n1kpc = scaringi1kpc_pop.shape[0]
    scar_n1kpc = 7284


    # LSS checking that whatever draw is made has 42 sources within 150 pc (after normalization) to match Pala sample.
    ind_check = []
    while len(ind_check) < scar_n1kpc:
        print("Drawing galactic positions...")
        xgalcent, ygalcent, zgalcent, inc = galactic_positions(overdraw, rng, model="McMillan_fixed",disk=disk)
        drawco = apyco.SkyCoord(x=xgalcent*u.kpc, y=ygalcent*u.kpc, z=zgalcent*u.kpc, frame='galactocentric', representation_type='cartesian')
        drawcoSSBc = drawco.transform_to(apyco.BarycentricMeanEcliptic)
        dist_from_sun = drawcoSSBc.distance.to(u.kpc).value
        
        # LSS astropy galactocentric coordinates assume 8.122 kpc for distance to gal center (origin of galcent)
        # LSS and z_sun = 20.8 pc. Sun is along x-axis. Lets find all systems within 1kpc of sun.
        draw_n1kpc = np.sum(dist_from_sun < 1)

        # LSS get normalizing factor and downsample the drawn pop to an accurate density of CV systems
        normfactor = scar_n1kpc / draw_n1kpc
        downsamp_ind = rng.choice(overdraw, int(overdraw*normfactor), replace=False)
        xgalcent = xgalcent[downsamp_ind]
        ygalcent = ygalcent[downsamp_ind]
        zgalcent = zgalcent[downsamp_ind]
        inc = inc[downsamp_ind]
        # LSS recalc distance post downsample
        drawco = apyco.SkyCoord(x=xgalcent*u.kpc, y=ygalcent*u.kpc, z=zgalcent*u.kpc, frame='galactocentric', representation_type='cartesian')
        drawcoSSBc = drawco.transform_to(apyco.BarycentricMeanEcliptic)
        dist_from_sun = drawcoSSBc.distance.to(u.kpc).value
        ind_check, = np.where(dist_from_sun<1)
        print('Not enough sources within 1 kpc, redrawing population...')

    drawco = apyco.SkyCoord(x=xgalcent*u.kpc, y=ygalcent*u.kpc, z=zgalcent*u.kpc, frame='galactocentric', representation_type='cartesian')
    drawcoSSBc = drawco.transform_to(apyco.BarycentricMeanEcliptic)
    dist_from_sun = drawcoSSBc.distance.to(u.kpc).value
    
    draw_n1kpc = np.sum(dist_from_sun < 1)
    print(f"Number of sources within 1 kpc after downsampling: {draw_n1kpc}")
    print(f"Difference between draw and Scaringi 1kpc pop after downsampling: {draw_n1kpc - scar_n1kpc}")
    print(f'This is a {100*(draw_n1kpc - scar_n1kpc)/scar_n1kpc:.2f}% difference.')
    
    # sample the primary mass with normal distribution supplied by user
    m1 = rng.normal(loc=mu_m1, scale=sigma_m1, size=len(xgalcent))
    
    # get the orbital periods by sampling from the Pala+2020 table
    porb = sample_porb_from_Pala_2020(nCV=len(xgalcent), rng=rng)
    f_gw = 2/(porb * 3600) # this is simple because the binaries are circular and porb is in hrs

    # get the matching donor mass from the Knigge+2011 table
    m2 = calculate_m2_from_porb(porb)
    m2_err = rngen.normal(loc=0, scale=sigma_m2, size=len(xgalcent))
    m2 = m2 + m2_err
    Pala_reassign = np.zeros(len(xgalcent))
    scar_reassign = np.zeros(len(xgalcent))
    dat = np.vstack([m1, m2, f_gw, inc, xgalcent, ygalcent, zgalcent, scar_reassign, Pala_reassign, dist_from_sun]).T

    # next reassign some of the sources to match the Scaringi data exactly
    scaringi1kpc_pop = pd.read_csv(paths.data / "dat_maxDistance_1000_final.txt")
    # m1[Msun], m2[Msun], f_gw[Hz], inclination[rad], x_galcent[kpc], y_galcent[kpc], z_galcent[kpc], Pala_reassigned (1 if reassigned to match Pala, 2 if in 150pc sample but not reassigned, 0 otherwise)

    m1_S, m2_S, fgw_S, inc_S, x_S, y_S, z_S, P_r = scaringi1kpc_pop.iloc[:,:].values.T
    
    ind_1kpc, = np.where(dist_from_sun<1) # LSS finding the sources within 1 kpc for reassignment to Scaringi pop.
    print(len(ind_1kpc), "sources within 1 kpc available for reassignment to Scaringi sample.") 

    # Some hacking required here. Pala sample is 42 sources, so we need to randomly select 42 sources
    # from the 150pc sample and replace with the Pala sample.
    # But we also need to make sure that we don't replace the same source twice.
    ind_scar = rng.choice(ind_1kpc, len(m2_S), replace=False)   
    dat[ind_1kpc, 7] = 2*np.ones(len(ind_1kpc)) # LSS flagging any that are within 1kpc but do not get reassigned as 2.

    dat[ind_scar, 0] = m1_S
    dat[ind_scar, 1] = m2_S
    dat[ind_scar, 2] = fgw_S
    dat[ind_scar, 3] = inc_S
    dat[ind_scar, 4] = x_S
    dat[ind_scar, 5] = y_S
    dat[ind_scar, 6] = z_S
    dat[ind_scar, 7] = np.ones(len(m1_S)) # LSS any that got reassigned to scaringi data is 1
    dat[ind_scar, 8] = P_r # LSS same rules with Pala 42 sources / 150 pc -- 2=within 150 pc, 1=w/i 150pc and reassigned to a Pala source.

    scarco = apyco.SkyCoord(x=x_S*u.kpc, y=y_S*u.kpc, z=z_S*u.kpc, frame='galactocentric', representation_type='cartesian')
    scarcoSSBc = scarco.transform_to(apyco.BarycentricMeanEcliptic)
    scar_dist_from_sun = scarcoSSBc.distance.to(u.kpc).value
    dat[ind_scar, 9] = scar_dist_from_sun
    return dat

def convert_popsynth_to_blipreadable(input_loc, output_loc):
    """ Converts population file to blip readable format with columns f, h, lat, long.

    Parameters
    ----------
    input_loc : str
        Location of the input file containing the population synthesis data.
    output_name : str
        Name of the output file to save the converted data in blip readable format.
    """
    file = input_loc
    binaries = pd.read_csv(file)
    blip_columns = ['f','h','lat','long']
    xG, yG, zG = binaries[' x_galcent[kpc]'].to_numpy(), binaries[' y_galcent[kpc]'].to_numpy(), binaries[' z_galcent[kpc]'].to_numpy()
    ## convert to distances and lat/long ecliptic coords
    gc = apyco.SkyCoord(x=xG*u.kpc,y=yG*u.kpc,z=zG*u.kpc, frame='galactocentric')
    SSBc = gc.transform_to(apyco.BarycentricMeanEcliptic)

    ## get latitude, longitude
    lat = SSBc.lat.to(u.rad).value
    long = SSBc.lon.to(u.rad).value

    ## making sure we've handled our coordinate transforms correctly
    dist = SSBc.distance.to(u.kpc)
    mc = lw.utils.chirp_mass(binaries['# m1[Msun]'],binaries[' m2[Msun]']).to_numpy()*u.Msun
    fs = binaries[' f_gw[Hz]'].to_numpy()
    f_orb = fs*u.Hz/2
    ## assuming circular binaries
    ecc = np.zeros(len(f_orb))

    hs = lw.strain.h_0_n(mc,f_orb,ecc,2,dist)

    blip_df = pd.DataFrame(data=np.vstack((fs,hs.flatten(),lat,long)).T,columns=blip_columns)

    blip_df.to_csv(output_loc, index=False,sep=' ',header=False)

def cvpop_frequency_hist_noEMedge(cvpop, freqcolname=' f_gw[Hz]', plot=True, outdir=None):
    """
    Create a histogram of CV population in freq. avoid edge effects before EM gap.

    Parameters
    ----------
    cvpop : DataFrame
        DataFrame containing CV population data.
    freqcolname : str, optional
        Column name for frequency in the CV population DataFrame
        (default is ' f_gw[Hz]').

    Returns
    -------
    counts : array
        Array of counts in each frequency bin.
    bins : array
        Array of bin edges for the frequency histogram. shape N+1 for N bins.


    """
    # LSS bin up the CV population to find EM gap
    # LSS we don't want edge effects, so this little loop decreases the number of bins
    # LSS until the lowest freq bin before the EM gap has the most CVs before the gap.
    # LSS edge effects can create a step down into the EM gap that isn't super accurate.
    edgects = -1 # LSS get us started
    counts, bins = np.histogram(cvpop[freqcolname], bins='auto')
    nbns = len(bins)-1
    while edgects < 0:
        counts, bins = np.histogram(cvpop[freqcolname], bins=nbns)
        emgapinds = np.where(counts == 0)[0]
        edgects = counts[emgapinds[0]-1] - counts[emgapinds[0]-2]
        if edgects < 0:
            print('Edge effects found! removing 1 bin')
            nbns -= 1

    print('Final number of freq bins for CV population:', nbns)

    if plot:
        plt.stairs(counts, bins)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Count')
        plt.title('CV Population Frequency Distribution')
        if outdir is not None:
            plt.savefig(outdir + 'cvpop_freq_hist_noEMedge.png', dpi=300, bbox_inches='tight', format='png')
        else:
            plt.savefig(paths.lssfigs / 'cvpop_freq_hist_noEMedge.png', dpi=300, bbox_inches='tight', format='png')
        plt.close()

    return counts, bins



def dNdmchirp(cvpop, rseed, outdir, disk='thin', freqcolname=' f_gw[Hz]', mass1colname='# m1[Msun]', mass2colname=' m2[Msun]', plot=True):
    """
    Create chirp mass distribution from given CV population using KDE.

    Nbins is used to bin the CVs according to frequency, this function then 
    finds the frequency bin with the most CVs before the EM freq gap and 
    uses the CVs in that bin to form a chirp mass distribution using a KDE. This distribution
    is then returned as an interpolation function and can be used to fill the 
    missing CV population in the EM gap using the N_CVs_gwevol function.

    Parameters
    ----------
    cvpop : DataFrame
        DataFrame containing CV population data.
    rseed : int
        Random seed for reproducibility in KDE sampling.
    outdir : str
        Directory to save any output plots or files.
    disk : str, optional
        Whether to use the 'thin' or 'thick' disk population for the CVs (default is 'thin'). Used here only for naming output files.
    freqcolname : str, optional
        Column name for frequency in the CV population DataFrame 
        (default is ' f_gw[Hz]').
    mass1colname : str, optional
        Column name for primary mass in the CV population DataFrame
        (default is '# m1[Msun]').
    mass2colname : str, optional
        Column name for secondary mass in the CV population DataFrame
        (default is ' m2[Msun]').

    Returns
    -------
    mc_interp : scipy.interpolate.interp1d (function)
        Interpolation of chirp mass distribution at lowest frequency bin.
        to get value at mchirp_val, use mc_interp(mchirp_val).
    """
    counts, bins = cvpop_frequency_hist_noEMedge(cvpop, freqcolname, plot=False)

    df = bins[1]-bins[0]
    emgap_inds = np.where(counts==0)[0] # LSS get indices of EM gap

    # LSS 20260129 - getting f bin with most CVs before gap. 
    # some CVs may have detached already and are not visible in the lowest freq bin before EM gap.
    fprime_ind = np.argmax(counts[:emgap_inds[0]])
    fprime = bins[fprime_ind]
    
    # LSS get the CVs in fprime
    fprime_CVs_ind = np.where((cvpop[freqcolname]>bins[fprime_ind]) &
                    (cvpop[freqcolname]<bins[fprime_ind]+df))[0]
    
    #print('N CVs in fprime bin:', fprime_CVs_ind.shape[0])
    fprime_CVs = cvpop.iloc[fprime_CVs_ind]
    
    # LSS check that we recovered the right number of CVs
    if fprime_CVs_ind.shape[0] != counts[fprime_ind]:
        print(f'{fprime_CVs_ind.shape[0]=}, {counts[fprime_ind]=}')
        raise Exception('Number of CVs in fprime bin does not match histogram count!')

    fprime_mc = lw.utils.chirp_mass(fprime_CVs[mass1colname] * u.Msun, fprime_CVs[mass2colname] * u.Msun)

    plt.hist(fprime_mc, bins='auto', histtype='step', density=True, label=f'Chirp Mass Distribution at fprime: {fprime:.1e} Hz');

    kern = 'gaussian'
    
    print('Finding optimal bandwidth for KDE of chirp mass distribution at fprime bin...')
    bwrange = np.logspace(-5, -2, 100) 
    K = 5 # Do 20-fold cross validation
    grid = GridSearchCV(KernelDensity(kernel=kern), {'bandwidth': bwrange}, cv=K) 
    print('starting grid search for optimal bandwidth...')
    # LSS find optimal bandwidth based on chirp mass distribution at fprime
    grid.fit(np.array(fprime_mc)[:, None]) 
    h_opt = grid.best_params_['bandwidth']
    print(f"Optimal bandwidth for chirp mass KDE at lowest frequency bin before EM gap: {h_opt}")

    # LSS make KDE over range of chirp masses in fprime bin
    mc_range = np.linspace(min(fprime_mc), max(fprime_mc), 100)
    mc_kde = KernelDensity(kernel=kern, bandwidth=h_opt).fit(np.array(fprime_mc)[:, None])

    # LSS get log density values from KDE
    log_dens = mc_kde.score_samples(mc_range[:, None])

    # normalize and return 0 if chirp mass out of interpolation range
    mc_interp = interp1d(mc_range, np.exp(log_dens)/np.sum(np.exp(log_dens)), fill_value=0, bounds_error=False)
    
    if plot:
        plt.hist(fprime_mc, bins='auto', histtype='step', density=True, label=f'Chirp Mass Distribution at fprime: {fprime:.1e} Hz');
        plt.plot(mc_range, np.exp(log_dens), label=f'Gaussian KDE, BW={h_opt:.1e}')
        plt.title('CV Chirp Mass Distribution before EM Gap', fontsize='xx-large')
        plt.xlabel(r'Chirp Mass [$M_{\odot}$]')
        plt.legend()
        plt.savefig(outdir + f'cvkpcsn_chirp_mass_dist_kde_{disk}_rs{rseed}.png', dpi=300, bbox_inches='tight', format='png')
        plt.close()

    return mc_interp, mc_range


def N_CVs_gwevol(cvpop, rseed, outdir, disk='thin', freqcolname=' f_gw[Hz]', mass1colname=' m1[Msun]', mass2colname=' m2[Msun]', plot=True):
    """
    Calculate the number of CVs at given frequencies assuming only GW evolution.

    This function assumes the chirp mass distribution is the same as that at the
    frequency bin with the most CVs before the EM freq gap.

    Parameters
    ----------
    cvpop : DataFrame
        DataFrame containing CV population data.
    rseed : int
        Random seed for reproducibility.
    outdir : str
        Directory to save output files.
    disk : str, optional
        Whether to use 'thin' or 'thick' disk for galactic distribution (default is 'thin'). Here it doesn't really matter but is used for saving file names.
    freqcolname : str, optional
        Column name for frequency in the CV population DataFrame
        (default is ' f_gw[Hz]').
    mass1colname : str, optional
        Column name for primary mass in the CV population DataFrame
        (default is ' m1[Msun]').
    mass2colname : str, optional
        Column name for secondary mass in the CV population DataFrame
        (default is ' m2[Msun]').
    plot : bool, optional
        Whether to plot the results (default is True).

    Returns
    -------
    N : array
        number of CVs at each frequency (summed over chirp mass)
    """
    # LSS get freq binning of CV pop
    counts, fbins = cvpop_frequency_hist_noEMedge(cvpop, freqcolname, plot=False)

    # LSS determine frequency and chirp mass ranges over which to fill the 2D distribution
    emgap_inds = np.where(counts==0)[0] # LSS get indices of EM gap
    fprime_ind = np.argmax(counts[:emgap_inds[0]])
    # LSS here we want to include the first populated bin on high freq edge 
    # of the emgap in case there are edge effects at top of emgap and 
    # we need to fill into that region. So we take +1 (the filled high freq bin)
    # and +1 the right edge of that bin to make sure we include the whole bin 
    # in the filling region.
    fprime_and_fgap = fbins[fprime_ind:emgap_inds[-1]+2]
    dNdmc_func, mc_range = dNdmchirp(cvpop, rseed, outdir, disk=disk, freqcolname=freqcolname, mass1colname=mass1colname, mass2colname=mass2colname)

    # LSS get df and dm for converting later density to number
    df = fbins[1]-fbins[0]
    dm = mc_range[1]-mc_range[0]

    # LSS create meshgrid from input arrays
    Fgrid, Mgrid = np.meshgrid(fprime_and_fgap, mc_range)

    # See eqn A2 of Nissanke et al. 2012
    # G, c in m, s, kg
    dfdt_fact = (96/5)*np.pow(np.pi, -8/3)*np.pow(apyconst.G.value, -5/3)*np.pow(apyconst.c.value, 5)

    # LSS convert chirp mass from solar masses to kg
    mc_kg = Mgrid * apyconst.M_sun.value
    dfdt_vars = np.pow(mc_kg, 5/3)*np.pow(Fgrid, 11/3)
    dtdf = (dfdt_fact*dfdt_vars)**-1

    # LSS get chirp mass distribution based on input binning of CV population
    dNdmc = dNdmc_func(Mgrid)
    
    # LSS calculate normalization via number in fprime bin
    normfact = counts[fprime_ind] / np.sum(dtdf[:, 0] * dNdmc[:, 0] * df * dm)

    # LSS sanity check with normalization
    tstNfprime = np.sum(dtdf[:, 0] * dNdmc[:, 0] * normfact * df * dm)
    if np.round(tstNfprime) != counts[fprime_ind]:
        print(f'{tstNfprime=}, {counts[fprime_ind]=}')
        raise Exception("Normalization check failed: calculated number in fprime bin does not match histogram count!")

    fspace = fprime_and_fgap
    N = dtdf * dNdmc * normfact * df * dm
    np.save(outdir + f'N_gwevol_unsmoothed_{disk}_rs{rseed}.npy', N)
    if plot:
        plt.imshow(N, aspect='auto', origin='lower', extent=[fprime_and_fgap[0], fprime_and_fgap[-1],
                                                            mc_range[0], mc_range[-1]],)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Chirp Mass [$M_{\odot}$]')
        plt.title('N_gwevol Unsmoothed')
        plt.savefig(outdir + f'N_gwevol_unsmoothed_{disk}_rs{rseed}.png', dpi=300, bbox_inches='tight', format='png')
        plt.close()

    # LSS since this joint distribution could be coarser in frequency, smooth a bit 
    # with a linear interpolation
    if N.shape[1] < N.shape[0]:
        print('Smoothing N_gwevol in frequency with linear interpolation...')
        Ndist_finterp = np.zeros((N.shape[0], N.shape[0]))
        fspace = np.linspace(fprime_and_fgap[0], fprime_and_fgap[-1], 100)
        for a in range(N.shape[0]):
            fint = np.interp(fspace, fprime_and_fgap,  N[a,:])
            Ndist_finterp[a,:] = fint
        np.save(outdir + f'N_gwevol_smoothed_{disk}_rs{rseed}.npy', Ndist_finterp)
        if plot:
            plt.imshow(Ndist_finterp, aspect='auto', origin='lower', extent=[fspace[0], fspace[-1],
                                                            mc_range[0], mc_range[-1]],)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(r'Chirp Mass [$M_{\odot}$]')
            plt.title('N_gwevol Smoothed in Frequency')
            plt.savefig(outdir + f'N_gwevol_smoothed_{disk}_rs{rseed}.png', dpi=300, bbox_inches='tight', format='png')
            plt.close()
            
        N = Ndist_finterp

    # LSS lets normalize N to prepare for rejection sampling.
    N = N/np.sum(N)

    return N, fspace, mc_range

def rejection_sample_emgap(N, fspace, mc_range, seednum, nsamples, outdir, npar=1000, plot=True):
    """Simple rejection sampling to fill the EM gap based on the N_gwevol distribution.

    Parameters
    ----------
    N : array
        2D normalized joint distribution of number of CVs as a function of frequency and chirp mass.
    fspace : array
        array of frequencies corresponding to the columns of N. shape (Nf,)
    mc_range : array
        array of chirp masses corresponding to the rows of N. shape (Nmc,)
    seednum : int
        seed for random number generator
    nsamples : int
        number of samples to generate

    Returns
    -------
    truepts : array
        array of shape (nsamples, 2) containing the frequency and 
        chirp mass of the sampled points in the supplied EM gap distribution.
    """
    rng = np.random.default_rng(seednum)
    truepts = np.zeros((nsamples, 2))
    accepted = 0

    while accepted < nsamples:
        randmc = rng.uniform(low=mc_range[0], high=mc_range[-1], size=npar)
        randf = rng.uniform(low=fspace[0], high=fspace[-1], size=npar)
        randnum = rng.uniform(low=0, high=1, size=npar)
        f_idx = np.argmin(np.abs(fspace[:, None] - randf[None, :]), axis=0)
        m_idx = np.argmin(np.abs(mc_range[:, None] - randmc[None, :]), axis=0)
        stat = randnum < N[m_idx, f_idx]
        lo, hi = accepted, np.min([accepted+np.sum(stat), nsamples])
        truepts[lo:hi, 0] = (randf[stat])[:hi-lo]
        truepts[lo:hi, 1] = (randmc[stat])[:hi-lo]
        accepted += np.sum(stat)

    truepts = np.array(truepts)
    np.save(outdir + f'rejection_sampled_emgap_{seednum}_{nsamples:.1e}samp.npy', truepts)
    if plot:
        plt.scatter(truepts[:,0], truepts[:,1], s=1, c='w', alpha=0.5)
        plt.imshow(N, aspect='auto', origin='lower', extent=[fspace[0], fspace[-1], mc_range[0], mc_range[-1]], alpha=0.7)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Chirp Mass [$M_{\odot}$]')
        plt.title(f'Rejection Sampled EM Gap, {nsamples} samples')
        plt.savefig(outdir + f'rejection_sampled_emgap_{seednum}_{nsamples:.1e}samp.png', dpi=300, bbox_inches='tight', format='png')
        plt.close()

    return truepts

if __name__ == '__main__':
    
    ## THESE ARE FIXED FOR THIS STUDY
    #max_distance = 20000 # pc or 20 kpc, this is the radius of the MW
    mu_m1 = 0.7
    sigma_m1 = 0.001
    sigma_m2 = 0.001
    disc = 'zh280pc'
    numsamp = int(1e6)

    # FIX A SEED TO REPRODUCE THE SAMPLE
    rseed = 2035
    rngen = np.random.default_rng(rseed)
    outputdir = paths.lssdata / f'CV_1kpcsn/rseed{rseed}_{disc}/'
    outputdir = str(outputdir) + '/'
    print(f'Output datafiles and figures will be saved in: {outputdir}')
    os.makedirs(outputdir, exist_ok=True)

    file = paths.data / 'dat_maxDistance_1000_final.txt'
    pddat = pd.read_csv(file)

    # LSS now we need to fill the EM gap
    # first we find joint freq/chirp mass dist for CVs in emgap according to GR
    # LSS N is the joint probability distribution of CVs in the f, mc space through the EM gap.
    # LSS fspace and mc_range are the corresponding frequencies and chirp masses
    # for the rows and columns of N.
    N, fspace, mspace = N_CVs_gwevol(pddat, disk=disc, rseed=rseed, outdir=outputdir, freqcolname=' f_gw[Hz]', mass1colname='# m1[Msun]', mass2colname=' m2[Msun]')

    # rejection sample this distribution
    # LSS pts will have shape (numsamp, 2) with columns f and mchirp
    print('Rejection sampling EM gap...')
    pts = rejection_sample_emgap(N, fspace, mspace, outdir=outputdir, seednum=rseed, nsamples=numsamp, npar=int(1e6))

    # then we need to normalize according to the number of CVs in the 
    # lowest freq bin just before the EM gap.
    counts, bns = cvpop_frequency_hist_noEMedge(pddat, freqcolname=' f_gw[Hz]', plot=False)
    emg_cts, _ = np.histogram(pts[:, 0], bins=bns) # LSS getting counts of EM gap draws 
    num_normfactor = emg_cts/counts
    num_normfactor[~np.isfinite(num_normfactor)] = 0 # set inf values to 0
    num_normfactor = num_normfactor[np.where(num_normfactor > 0)[0][0]]

    # LSS how many pts do we need to draw to match the number in gxy
    normdraws = pts.shape[0]/num_normfactor 
    # LSS downsample
    dwnsamp_pts = pts[rngen.choice(pts.shape[0], int(normdraws), replace=True)]
    emgap_inds = np.where(counts==0)[0] 
    fprime_ind = np.argmax(counts[:emgap_inds[0]])
    fprime = bns[fprime_ind]
    df = bns[1]-bns[0]
    # LSS only keep points drawn within the EM gap (above fprime + df)
    dwnsamp_pts_emgap = dwnsamp_pts[dwnsamp_pts[:,0]>fprime+df]
    print(f"Number of points drawn in EM gap after downsampling: {len(dwnsamp_pts_emgap)}")
    plt.stairs(counts, bns, label='CV population')
    plt.hist(dwnsamp_pts[:, 0], bins=bns, histtype='step', label='Rejection Sampled Points')
    plt.hist(dwnsamp_pts_emgap[:, 0], bins=bns, histtype='step', label='Rejection Sampled Points in EM gap')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Number of CVs')
    plt.legend()
    plt.savefig(outputdir + f'cvkpcsn_emg_filled_rseed{rseed}_nsamp{numsamp:.1e}_freq_hist.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

    # then we need to assign galactic positions to CVs in EM gap.
    max_dist = 1000 # u.kpc

    x_gal, y_gal, z_gal = sample_position_from_Pala_2020(rng=rngen, dist_max=1000)
    x_gal = x_gal[:len(dwnsamp_pts_emgap)]
    y_gal = y_gal[:len(dwnsamp_pts_emgap)]
    z_gal = z_gal[:len(dwnsamp_pts_emgap)]
    emgap_gal_coord = apyco.SkyCoord(x_gal*u.kpc, y_gal*u.kpc, z_gal*u.kpc, frame='galactic', representation_type='cartesian')
    inc = np.pi - np.arccos(rngen.uniform(-1, 0, len(dwnsamp_pts_emgap))) # LSS random inclinations for EM gap CVs, uniform in cos(inc)

    # LSS plot a comparison skymap of positions
    cvkpcsn_galcentco = apyco.SkyCoord(x=pddat[' x_gal[kpc]'].to_numpy()*u.kpc, y=pddat[' y_gal[kpc]'].to_numpy()*u.kpc, z=pddat[' z_gal[kpc]'].to_numpy()*u.kpc, frame='galactocentric', representation_type='cartesian')
    emgap_galcentco = emgap_gal_coord.transform_to(apyco.Galactocentric)
    cvkpcsn_SSBc = cvkpcsn_galcentco.transform_to(apyco.BarycentricMeanEcliptic)
    emgap_SSBc = emgap_galcentco.transform_to(apyco.BarycentricMeanEcliptic)
    cvkpcsn_lat = cvkpcsn_SSBc.lat.to(u.rad).value
    cvkpcsn_lon = cvkpcsn_SSBc.lon.to(u.rad).value
    emgap_lat = emgap_SSBc.lat.to(u.rad).value
    emgap_lon = emgap_SSBc.lon.to(u.rad).value
    hp.mollview(coord='E')
    hp.projscatter(cvkpcsn_lon*u.rad.to(u.deg),cvkpcsn_lat*u.rad.to(u.deg),lonlat=True,color='k',alpha=0.5,s=4, label='CV Population')
    hp.projscatter(emgap_lon*u.rad.to(u.deg),emgap_lat*u.rad.to(u.deg),lonlat=True,color='r',alpha=0.5,s=10, label='EM gap CVs')
    plt.legend()
    plt.savefig(outputdir + f'cvkpcsn_emg_rseed{rseed}_nsamp{numsamp}_skymap.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

    # LSS save emgap CVs to a file
    emgap_dat = np.hstack((dwnsamp_pts_emgap, x_gal[:, None], y_gal[:, None], z_gal[:, None], inc[:, None]))
    np.savetxt(outputdir + f"dat_kpcsn_emgap_rs{rseed}_nsamp{numsamp:.1e}_final.txt", emgap_dat, delimiter=',', header="f_gw[Hz], Mchirp[Msun], x_gal[kpc], y_gal[kpc], z_gal[kpc], inclination[rad]", fmt='%.10f')

    # LSS convert everything to a BLIP readable format.
    blip_columns = ['f','h','lat','long']

    ## making sure we've handled our coordinate transforms correctly
    cvkpcsn_dist = cvkpcsn_SSBc.distance.to(u.kpc)
    emgap_dist = emgap_SSBc.distance.to(u.kpc)
    cvkpcsn_mc = lw.utils.chirp_mass(pddat['# m1[Msun]'], pddat[' m2[Msun]']).to_numpy()*u.Msun
    emgap_mc = dwnsamp_pts_emgap[:, 1]*u.Msun # LSS this is already the chirp mass
    cvkpcsn_fs = pddat[' f_gw[Hz]'].to_numpy()
    cvkpcsn_f_orb = cvkpcsn_fs*u.Hz/2
    emgap_fs = dwnsamp_pts_emgap[:, 0]
    emgap_f_orb = emgap_fs*u.Hz/2
    ## assuming circular binaries
    cvkpcsn_ecc = np.zeros(len(cvkpcsn_f_orb))
    emgap_ecc = np.zeros(len(emgap_f_orb))
    cvkpcsn_hs = lw.strain.h_0_n(cvkpcsn_mc,cvkpcsn_f_orb,cvkpcsn_ecc,2,cvkpcsn_dist)
    emgap_hs = lw.strain.h_0_n(emgap_mc,emgap_f_orb,emgap_ecc,2,emgap_dist)

    cvkpcsn_blip_df = pd.DataFrame(data=np.vstack((cvkpcsn_fs,cvkpcsn_hs.flatten(),cvkpcsn_lat,cvkpcsn_lon)).T,columns=blip_columns)
    emgap_blip_df = pd.DataFrame(data=np.vstack((emgap_fs,emgap_hs.flatten(),emgap_lat,emgap_lon)).T,columns=blip_columns)
    cvkpcsn_blip_df.to_csv(outputdir + f"dat_kpcsn_rs{rseed}_nsamp{numsamp:.1e}_BLIP_final.txt", index=False,sep=' ',header=False)
    emgap_blip_df.to_csv(outputdir + f"dat_kpcsn_emgap_rs{rseed}_nsamp{numsamp:.1e}_BLIP_final.txt", index=False,sep=' ',header=False)

    combined_fs = np.hstack((cvkpcsn_fs, emgap_fs))
    combined_hs = np.hstack((cvkpcsn_hs.flatten(), emgap_hs.flatten()))
    combined_lat = np.hstack((cvkpcsn_lat, emgap_lat))
    combined_lon = np.hstack((cvkpcsn_lon, emgap_lon))
    cvkpcsn_emgap_blip_df = pd.DataFrame(data=np.vstack((combined_fs,combined_hs,combined_lat,combined_lon)).T,columns=blip_columns)
    cvkpcsn_emgap_blip_df.to_csv(outputdir + f"dat_kpcsn_emgap_combined_{disc}_rs{rseed}_nsamp{numsamp:.1e}_BLIP_final.txt", index=False,sep=' ',header=False)
