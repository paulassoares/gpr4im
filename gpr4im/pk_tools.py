import numpy as np


'''
This script contains Steve's MultipoleExpansion code for calculating the power spectrum.
'''


def getpk(datgrid,w,W,nx,ny,nz,lx,ly,lz,kbins):
    '''
    Obtain the 3D angle-averaged power spectrum for auto-correlation.
    
    ------
    Inputs
    ------
    
    datgrid: data cube you wish to measure the power spectrum of, shape [Nx, Ny, Nz]
    w: weighting function (e.g. inverse noise weighting), shape [Nx, Ny, Nz]
    W: pixel masking function (1 for pixels to consider, 0 for pixels to ignore, shape [Nx, Ny, Nz]
    nx, ny, nz: shape of data cube [Nx, Ny, Nz]
    lx, ly, lz: data cube box dimensions, in Mpc/h
    kbins: edges of k_bins you wish to average your power spectrum into
    
    -------
    Outputs
    -------
    
    pk: 3D angle-averaged power spectrum
    nmodes: number of modes in each k bin
    
    '''
    vol = lx*ly*lz
    nc = nx*ny*nz
    pkspec = getpkspec(datgrid,nc,vol,w,W)
    pk, nmodes = binpk(pkspec,nx,ny,nz,lx,ly,lz,kbins,FullPk=False,doindep=True)
    return pk, nmodes


def getpkspec(datgrid,nc,vol,w,W):
    '''
    Estimate the 3D power spectrum of a density field (auti-correlation).
    '''
    Vcell = vol/nc
    Wmean = np.mean(W)
    W = W/np.sum(W)
    w = w/np.sum(w)
    fgrid = np.fft.rfftn(w * datgrid)
    pkspec = np.real( fgrid * np.conj(fgrid) )
    return vol * pkspec / ( nc*np.sum(W**2 * w**2) * Wmean**2 * nc**2 )


def binpk(pkspec,nx,ny,nz,lx,ly,lz,kbins,FullPk=False,doindep=True):
    '''
    Bin 3D power spectrum in angle-averaged bins.
    '''
    #print('\nBinning 3D power spectrum into angle-averaged bins...')
    kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz,FullPk)
    if doindep==True:
        pkspec = pkspec[indep==True]
        kspec = kspec[indep==True]
    ikbin = np.digitize(kspec,kbins)
    nkbin = len(kbins)-1
    nmodes,pk = np.zeros(nkbin,dtype=int),np.zeros(nkbin)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin==ik+1])))
        if (nmodes[ik] > 0): #if nmodes==0 for this k then remains Pk=0
            pk[ik] = np.mean(pkspec[ikbin==ik+1])
    return pk,nmodes


def get_biascorr(fg_cov, Input, n_samples, lx, ly, lz, kbins_par, kbins, w, W):
    '''
    Obtain random samples given our foreground covariance
    
    ------
    Inputs
    ------
    
    fg_cov: your foreground covariance (shape [Nfreq, Nfreq]), the error on your foreground fit
    Input: your input data, that you are trying to foreground clean, of shape [Nx, Ny, Nz]
    n_samples: number of samples you want to draw, each sample will have shape [Nx, Ny, Nz]
    lx, ly, lz: dimensions of your data cube, in Mpc/h
    w: weighting function
    W: window function
    
    -------
    Outputs
    -------
    
    pk_corr: bias correction for 3D P(k)
    pk_corr_para: bias correction for 1D P(k_parallel)
    samples: an array of samples derived from fg_cov, total of n_samples samples, each of
        shape [Nx, Ny, Nz]
    '''
    nx, ny, nz = Input.shape
    mu = np.zeros(nz) # vector of the means, zero

    # Generate n separate samples paths from a Gaussian with mean mu and covariance fg_cov:
    samples = []
    for i in range(n_samples):
        s = np.random.multivariate_normal(mu, fg_cov, (nx, ny))# * Input.std() * noise_smoothed.std()
        samples.append(s)
    samples = np.array(samples)
    
    # Obtain 3D P(k) and 1D P(k_para) for each of these samples
    pk_corr = []
    pk_corr_para = []
    for s in samples:
        pk_corr_para.append(ParaPk(s, nx, ny, nz, lx, ly, lz, kbins_par, w, W)[0])
        pk_corr.append(getpk_noPerp(s,w,W,nx,ny,nz,lx,ly,lz,kbins)[0])
    
    # Average power spectral over all samples:
    pk_corr = np.mean(pk_corr, axis=0)
    pk_corr_para = np.mean(pk_corr_para, axis=0)
    
    return pk_corr, pk_corr_para, samples


def PerpParaPk(datgrid,nx,ny,nz,lx,ly,lz,kperpbins,kparabins,w,W):
    '''
    Return 2D image of P(k_perp,k_para)
    '''
    # Obtain two 3D arrays specifying kperp and kpara values at every point in
    #    pkspec array
    print('\nCalculating P(k_perp,k_para)...')
    pkspec = getpkspec(datgrid,nx*ny*nz,lx*ly*lz,w,W)
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kperp = np.sqrt(kx[:,np.newaxis]**2 + ky[np.newaxis,:]**2)
    kpara = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    kperp_arr = np.reshape( np.repeat(kperp,int(nz/2)+1) , (nx,ny,int(nz/2)+1) )
    kpara_arr = np.tile(kpara,(nx,ny,1))
    # Identify and remove non-independent modes
    null1,null2,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep==True]
    kperp_arr = kperp_arr[indep==True]
    kpara_arr = kpara_arr[indep==True]
    # Get indices where kperp and kpara values fall in bins
    ikbin_perp = np.digitize(kperp_arr,kperpbins)
    ikbin_para = np.digitize(kpara_arr,kparabins)
    nkperpbin,nkparabin = len(kperpbins)-1,len(kparabins)-1
    nmodes,pk2d = np.zeros((nkparabin,nkperpbin),dtype=int),np.zeros((nkparabin,nkperpbin))
    for i in range(nkperpbin):
        for j in range(nkparabin):
            nmodes[j,i] = int(np.sum(np.array([(ikbin_perp==i+1) & (ikbin_para==j+1)])))
            if (nmodes[j,i] > 0):
                # Average power spectrum into (kperp,kpara) cells
                pk2d[j,i] = np.mean(pkspec[(ikbin_perp==i+1) & (ikbin_para==j+1)])
    return pk2d,nmodes


def ParaPk(datgrid, nx, ny, nz, lx, ly, lz, kparabins, w, W):
    '''
    Return 1D P(k_para)
    '''
    # calculating power spectrum of overdensity grid:
    pkspec = getpkspec(datgrid,nx*ny*nz,lx*ly*lz,w,W)
    # get 1D array of available k_para modes
    kpara = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    # fill a cube the size of our power spectrum with copies of kpara
    kpara_arr = np.tile(kpara,(nx,ny,1))
    # identify and remove non-independent modes
    null1,null2,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep==True]
    kpara_arr = kpara_arr[indep==True]
    # get indices where kpara values fall in desired bins
    ikbin_para = np.digitize(kpara_arr,kparabins)
    nkparabin = len(kparabins)-1
    nmodes1d,pk1d = np.zeros(nkparabin,dtype=int),np.zeros(nkparabin)
    for i in range(nkparabin):
        nmodes1d[i] = int(np.sum(np.array([ikbin_para==i+1])))
        if (nmodes1d[i] > 0):
            # average power spectrum into kpara cells
            pk1d[i] = np.mean(pkspec[ikbin_para==i+1])
    return pk1d, nmodes1d


def PerpPk(datgrid,nx,ny,nz,lx,ly,lz,kperpbins,w,W):
    '''
    Return 1D P(k_perp)
    '''
    pkspec = getpkspec(datgrid,nx*ny*nz,lx*ly*lz,w,W)
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kperp = np.sqrt(kx[:,np.newaxis]**2 + ky[np.newaxis,:]**2)
    kperp_arr = np.reshape( np.repeat(kperp,int(nz/2)+1) , (nx,ny,int(nz/2)+1) )
    # Identify and remove non-independent modes
    null1,null2,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep==True]
    kperp_arr = kperp_arr[indep==True]
    # Get indices where kperp and kpara values fall in bins
    ikbin_perp = np.digitize(kperp_arr,kperpbins)
    nkperpbin = len(kperpbins)-1
    nmodes1d,pk1d = np.zeros((nkperpbin),dtype=int),np.zeros((nkperpbin))
    for i in range(nkperpbin):
        nmodes1d[i] = int(np.sum(np.array([ikbin_perp==i+1])))
        if (nmodes1d[i] > 0):
            # Average power spectrum into kperp cells
            pk1d[i] = np.mean(pkspec[ikbin_perp==i+1])
    return pk1d, nmodes1d


def getpk_noPerp(datgrid,w,W,nx,ny,nz,lx,ly,lz,kbins):
    '''
    Return 1D P(k_perp)
    '''
    pkspec = getpkspec(datgrid,nx*ny*nz,lx*ly*lz,w,W)
    # get 1D array of available k_para modes
    kpara = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    # fill a cube the size of our power spectrum with copies of kpara
    kpara_arr = np.tile(kpara,(nx,ny,1))
    # identify and remove non-independent modes
    kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep==True]
    kpara_arr = kpara_arr[indep==True]
    kspec = kspec[indep==True]
    # get indices where kpara values fall in desired 1D bins
    nkbin = len(kbins)-1
    ikbin_para = np.digitize(kpara_arr,kbins)
    nmodes1d,pk1d = np.zeros(nkbin,dtype=int),np.zeros(nkbin)
    # same but for 3D bins:
    ikbin = np.digitize(kspec,kbins)
    nmodes3d,pk3d = np.zeros(nkbin,dtype=int),np.zeros(nkbin)
    for i in range(nkbin):
        nmodes1d[i] = int(np.sum(np.array([ikbin_para==i+1])))
        nmodes3d[i] = int(np.sum(np.array([ikbin==i+1])))
        if (nmodes1d[i] > 0):
            # average power spectrum into kpara cells
            pk1d[i] = np.mean(pkspec[ikbin_para==i+1])
    pk3d = (pk1d*nmodes1d)/nmodes3d
    return pk3d, nmodes3d


def getpk_2Dto3D(pk2d, nmodes2d, kpara_array, kperp_array, kbins):
    k2D = np.zeros((len(kpara_array), len(kperp_array)))
    pk3d = np.zeros(len(kbins)-1)
    nmodes = np.zeros(len(kbins)-1)
    for i in range(len(kperp_array)):
        for j in range(len(kpara_array)):
            k2D[j,i] = np.sqrt(kpara_array[j]**2 + kperp_array[i]**2)
    for i in range(len(kbins)-1):
        kmin, kmax = kbins[i], kbins[i+1]
        pk_i = []
        nmodes_i = []
        for k in range(len(kperp_array)):
            for j in range(len(kpara_array)):
                if (k2D[j,k] > kmin) & (k2D[j,k] < kmax):
                    if nmodes2d[j,k] > 0:
                        pk_i.append(pk2d[j,k]*nmodes2d[j,k])
                        nmodes_i.append(nmodes2d[j,k])
        pk_i = np.sum(pk_i)
        nmodes_i = np.sum(nmodes_i)
        pk3d[i] = pk_i/nmodes_i
        nmodes[i] = nmodes_i
    return pk3d, nmodes


def getkspec(nx,ny,nz,lx,ly,lz,FullPk=False):
    kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    if FullPk==True: kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)
    else: kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    indep = getindep(nx,ny,nz)
    indep[0,0,0] = False
    if FullPk==True:
        indep = fthalftofull(nx,ny,nz,indep)
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    kspec[0,0,0] = 1.
    muspec = np.absolute(kz[np.newaxis,np.newaxis,:])/kspec
    kspec[0,0,0] = 0.
    return kspec,muspec,indep


def fthalftofull(nx,ny,nz,halfspec):
    fullspec = np.empty((nx,ny,nz))
    ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(int(nz/2)+1,nz)
    ixneg[0],iyneg[0] = 0,0
    fullspec[:,:,:int(nz/2)+1] = halfspec
    fullspec[:,:,int(nz/2)+1:nz] = fullspec[:,:,izneg][:,iyneg][ixneg]
    return fullspec


def getindep(nx,ny,nz):
    indep = np.full((nx,ny,int(nz/2)+1),False,dtype=bool)
    indep[:,:,1:int(nz/2)] = True
    indep[1:int(nx/2),:,0] = True
    indep[1:int(nx/2),:,int(nz/2)] = True
    indep[0,1:int(ny/2),0] = True
    indep[0,1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),1:int(ny/2),0] = True
    indep[int(nx/2),1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),0,0] = True
    indep[0,int(ny/2),0] = True
    indep[int(nx/2),int(ny/2),0] = True
    indep[0,0,int(nz/2)] = True
    indep[int(nx/2),0,int(nz/2)] = True
    indep[0,int(ny/2),int(nz/2)] = True
    indep[int(nx/2),int(ny/2),int(nz/2)] = True
    return indep


