import numpy as np


'''
This script contains Steve's MultipoleExpansion code for calculating the power spectrum.
'''


def getpk(datgrid,w,W,nx,ny,nz,lx,ly,lz,kbins):
    '''
    Obtain the 3D angle-averaged power spectrum for auto-correlation.
    '''
    vol = lx*ly*lz
    nc = nx*ny*nz
    pkspec = getpkspec(datgrid,datgrid,nc,vol,w,W)
    pk, nmodes = binpk(pkspec,nx,ny,nz,lx,ly,lz,kbins,FullPk=False,doindep=True)
    return pk, nmodes


def getpkspec(datgrid1,datgrid2,nc,vol,w,W):
    '''
    Estimate the 3D power spectrum of a density field.
    If auto-correlating then datgrid1 should equal datgrid2
    '''
    Vcell = vol/nc
    Wmean = np.mean(W)
    W = W/np.sum(W)
    w = w/np.sum(w)
    fgrid1 = np.fft.rfftn(w * datgrid1)
    fgrid2 = np.fft.rfftn(w * datgrid2)
    pkspec = np.real( fgrid1 * np.conj(fgrid2) )
    #return vol * pkspec / (sumWwsq*(Wmean**2)*(nc**2))
    return vol * pkspec / ( nc*np.sum(W**2 * w**2) * Wmean**2 * nc**2 )


def PerpParaPk(datgrid,nx,ny,nz,lx,ly,lz,kperpbins,kparabins,w,W):
    '''
    Return 2D image of P(k_perp,k_para)
    '''
    # Obtain two 3D arrays specifying kperp and kpara values at every point in
    #    pkspec array
    print('\nCalculating P(k_perp,k_para)...')
    pkspec = getpkspec(datgrid,datgrid,nx*ny*nz,lx*ly*lz,w,W)
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
    pkspec = getpkspec(datgrid,datgrid,nx*ny*nz,lx*ly*lz,w,W)
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
    pkspec = getpkspec(datgrid,datgrid,nx*ny*nz,lx*ly*lz,w,W)
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
    pkspec = getpkspec(datgrid,datgrid,nx*ny*nz,lx*ly*lz,w,W)
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kperp = np.sqrt(kx[:,np.newaxis]**2 + ky[np.newaxis,:]**2)
    kperp_arr = np.reshape( np.repeat(kperp,int(nz/2)+1) , (nx,ny,int(nz/2)+1) )
    # Identify and remove non-independent modes
    kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep==True]
    kperp_arr = kperp_arr[indep==True]
    kspec = kspec[indep==True]
    # Get indices where kperp values fall in bins
    ikbin_perp = np.digitize(kperp_arr,kbins)
    nkperpbin = len(kbins)-1
    nmodes1d,pk1d = np.zeros((nkperpbin),dtype=int),np.zeros((nkperpbin))
    for i in range(nkperpbin):
        nmodes1d[i] = int(np.sum(np.array([ikbin_perp==i+1])))
        if (nmodes1d[i] > 0):
            # Set power spectrum on kperp cells to zero:
            pkspec[ikbin_perp==i+1] = 0
    ikbin = np.digitize(kspec,kbins)
    nkbin = len(kbins)-1
    nmodes,pk3d = np.zeros(nkbin,dtype=int),np.zeros(nkbin)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin==ik+1])))
        if (nmodes[ik] > 0): #if nmodes==0 for this k then remains Pk=0
            pk3d[ik] = np.mean(pkspec[ikbin==ik+1])
    return pk3d, nmodes


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


