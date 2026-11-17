"""Read the generated ISC files with torchvic.vicwur readers (run from any directory)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from torchvic.vicwur import params as P, forcing as F
A=os.environ.get('TORCHVIC_DATA_ROOT', 'data')
O=A+'/vicwur/isc'
g=P.Grid(A+'/vic_params_isc/domain.nc',O+'/params/params_wur.nc')
print('cells',g.n)
cv=P.read_cv(O+'/params/params_wur.nc',g);print(cv.shape)
irr=P.read_irrigation(O+'/params/irr.nc',g,cv,pbare=20,nbare=1) if hasattr(P,'read_irrigation') else None
print(type(irr))
b=P.read_bco2(O+'/params/params_wur.nc',g);print(b.shape)
for p,v in (('cv.2005','cv'),('mun_demand.2005','mun_demand'),('co2.2005','co2'),('liv_gw.2005','liv_gw')):
    print(p,F.file_info('%s/forcing/%s.nc'%(O,p),v), F.read_year('%s/forcing/%s.nc'%(O,p),v).shape)
