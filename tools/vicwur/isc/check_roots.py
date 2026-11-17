import netCDF4 as nc,numpy as np,sys
d=nc.Dataset(sys.argv[1]);cv=np.ma.filled(d['Cv'][:],0);rf=np.ma.filled(d['root_fract'][:],0);rd=np.ma.filled(d['root_depth'][:],0)
n=bad=nz=0
for v in range(cv.shape[0]-1):
    s=rf[v].sum(0);veg=cv[v]>0
    n+=int((veg&(s<=0)).sum());bad+=int((veg&(abs(s-1)>1e-6)).sum());nz+=int((veg&(rd[v].sum(0)<=0)).sum())
print('noroot',n,'sum!=1',bad,'zero depth',nz)
