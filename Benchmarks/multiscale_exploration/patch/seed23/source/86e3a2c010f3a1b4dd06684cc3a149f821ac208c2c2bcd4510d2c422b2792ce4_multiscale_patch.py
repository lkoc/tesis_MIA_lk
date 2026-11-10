"""Exploratory global/local soil enrichment for a known smooth conductivity patch."""
import torch
from torch import nn
from Benchmarks.multiscale_trace import TracePINN


class TracePatchPINN(TracePINN):
    def __init__(self, domain, variant='trace_patch', width=32, depth=3):
        super().__init__(domain, variant, width, depth)
        patch = domain.case.get('patch')
        if patch is None or patch[4] == domain.case['k']:
            raise ValueError('Patch enrichment requires a nontrivial smooth known conductivity patch')
        self.variant = 'trace_patch'
        layers=[]
        nin=4
        for _ in range(2):
            linear=nn.Linear(nin,16)
            nn.init.xavier_normal_(linear.weight);nn.init.zeros_(linear.bias)
            layers.extend([linear,nn.Tanh()]);nin=16
        last=nn.Linear(16,1)
        nn.init.xavier_normal_(last.weight);nn.init.zeros_(last.bias)
        layers.append(last)
        # Additional expert has no physical region of its own. It contributes
        # to G inside the same soil PDE, and its output coefficients enter SVD.
        self.networks.append(nn.Sequential(*layers))

    def global_soil(self, region, xy):
        base=super().global_soil(region,xy)
        c=self.domain.case
        xp,yp,wx,wy,kp,_=c['patch']
        u=(xy[:,:1]-xp)/wx;v=(xy[:,1:2]-yp)/wy
        delta=xy[:,:2]-xy.new_tensor(c['cables'][0])
        log_radius=torch.log(torch.linalg.vector_norm(delta,dim=1,keepdim=True)/c['radius'])
        contrast=(self.domain.conductivity(region,xy,torch)-c['k'])/(kp-c['k'])
        features=torch.cat([u,v,log_radius,contrast],dim=1)
        window=torch.exp(-.5*(u*u+v*v))
        return base+c['scale']*self.envelope(xy)*window*self.networks[-1](features)
