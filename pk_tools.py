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


