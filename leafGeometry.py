#!/usr/bin/python
""" leafGeometry.py

Implements various leaf inclination functions for
use in the sellersTwoStream code.
    
Copyright (C) 2016 Tristan Quaife

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Tristan Quaife
tquaife@gmail.com
"""
import sys

import numpy as np
import scipy.integrate as integrate

import matplotlib.pyplot as plt

#import warnings
#warnings.filterwarnings("error")

class leafGeometry( ):

  def __init__(self):
    """This class implements various leaf geometry functions.

    Currently contains:
    
    G function as implemented in JULES (using analytical solutions 
    to a small number of distributions)
    
    G function as implemented in CLM (Goudrian-type parameterisation)
    """
    self.CLM_chiL=0.01
    self.JULES_lad='uniform'
    
    self.gDash=self.gDash_bunnik_spherical

  def gDash(self, mu):
    pass    

  def gDash_bunnik_spherical(self, mu):
    """Spherical distribution from Bunnik.
    This is the same as uniform in other contexts.
    """
    theta=np.arccos(mu)
    return np.sin(theta)

  def gDash_bunnik_planophile(self, mu):
    """Planophile distribution from Bunnik.
    """
    theta=np.arccos(mu)
    return 2./np.pi*(1+np.cos(2*theta))

  def gDash_bunnik_erectophile(self, mu):
    """Erectophile distribution from Bunnik.
    """
    theta=np.arccos(mu)
    return 2./np.pi*(1-np.cos(2*theta))

  def gDash_bunnik_plagiophile(self, mu):
    """Plagiophile distribution from Bunnik.
    """
    theta=np.arccos(mu)
    return 2./np.pi*(1-np.cos(4*theta))

  def gDash_bunnik_extremophile(self, mu):
    """Extremophile distribution from Bunnik.
    """
    theta=np.arccos(mu)
    return 2./np.pi*(1+np.cos(4*theta))


  def rossPsi(self, muL, mu):
    """Ross Psi function for azimuthally unifrom distributions.
    Allows the G function to be computed using only a single integral
    from 0-pi/2 of gDash*rossPsi.
    
    Main form of Psi taken from Knyazikhin, Myneni and Stenberg (2004)
    additional checks taken from SemiDescrete code by Gobron et al. (1997)
    """
    theta=np.arccos(mu)
    thetaL=np.arccos(muL)
  
    #Preliminary checks:
    if muL==1.:
      return mu
    if np.sin(theta)==0.:
      return muL
    if np.sin(thetaL)==0:
      testVal=0.0
    else: 
      testVal=mu*muL
  
    #The Ross Psi function:
    if testVal>=(np.sin(theta)*np.sin(thetaL)):
      psi=np.abs(mu*muL)
    else:
      cotProduct=-1./np.tan(theta)*1./np.tan(thetaL)
      
      #need this to catch some numerical errors:
      #(cotProduct is sometimes very very slightly >1.)
      if cotProduct<=-1.0:
        branchAngle=np.pi
      else:
        branchAngle=np.arccos(cotProduct)
      
      psi=mu*muL*(2*branchAngle/np.pi-1.)
      psi+=2./np.pi*np.sqrt(1-mu*mu)*np.sqrt(1-muL*muL)*np.sin(branchAngle)
  
    return psi
  

  def gDash_rossPsi(self, thetaL, mu):  
    """ Method to be integrated to find G from gDash
    """
    muL=np.cos(thetaL)
    return self.gDash(muL)*self.rossPsi(muL, mu)
    

  def G_integ_gDash(self, mu):
    """Calculate the G function by integrating gDash*rossPsi
    """
    intg=integrate.quad(self.gDash_rossPsi,0,np.pi/2.,args=(mu,))      
    return intg[0]



  def G_uniform(self,mu):
    return 0.5

  def G_horizontal(self,mu):
    return mu


  def G_JULES(self,mu):
    if self.JULES_lad=='uniform':
      return self.G_uniform(mu)
    elif self.JULES_lad=='horizontal':
      return self.G_horizontal(mu)
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad
      
      
      

  def G_CLM(self,mu):
    """Calculate the G function as used in CLM
    
    Depends on parameter self.CLM_chiL
    
    Valid range of chiL is apparently -0.4 to 0.6    
    (although some of CLMs PFT exceed this range).
    See page 24 of CLM TN v3.0
    """
  
    if self.CLM_chiL < -1.0 or self.CLM_chiL > 1.0:
      raise Exception, "parameter chiL out of range: "%self.CLM_chiL

    return self.CLM_phi1()+self.CLM_phi2()*mu


  def CLM_phi1(self):
    """Required for CLM G function
    """
    return 0.5-0.633*self.CLM_chiL-0.33*self.CLM_chiL*self.CLM_chiL

  def CLM_phi2(self):
    """Required for CLM G function
    """
    return 0.877*(1-2.*self.CLM_phi1())



def test_GFunctions( ):

  l=leafGeometry()

  GFuncs={}
  GFuncs['Spherical']=l.gDash_bunnik_spherical
  GFuncs['Planophile']=l.gDash_bunnik_planophile
  GFuncs['Erectophile']=l.gDash_bunnik_erectophile
  GFuncs['Plagiophile']=l.gDash_bunnik_plagiophile
  GFuncs['Extremophile']=l.gDash_bunnik_extremophile

  for func in GFuncs:
  
    l.gDash=GFuncs[func]
    x=[]
    y=[]
    for theta in xrange(0,90,2):
      mu=np.cos(np.deg2rad(theta))
      x.append(theta)
      y.append(l.G_integ_gDash(mu))
    
    plt.plot(x,y,label=func)
  
  plt.xlabel('zenith angle (degrees)')
  plt.ylabel('g')
  plt.legend()
  plt.show()  


def test_gFunctions( ):

  l=leafGeometry()

  GFuncs={}
  GFuncs['Spherical']=l.gDash_bunnik_spherical
  GFuncs['Planophile']=l.gDash_bunnik_planophile
  GFuncs['Erectophile']=l.gDash_bunnik_erectophile
  GFuncs['Plagiophile']=l.gDash_bunnik_plagiophile
  GFuncs['Extremophile']=l.gDash_bunnik_extremophile

  for func in GFuncs:
  
    x=[]
    y=[]
    for theta in xrange(0,90,2):
      mu=np.cos(np.deg2rad(theta))
      x.append(theta)
      y.append(GFuncs[func](mu))
    
    plt.plot(x,y,label=func)
  
  plt.xlabel('zenith angle (degrees)')
  plt.ylabel('G')
  plt.legend()
  plt.show()  



if __name__=="__main__":

  test_gFunctions()  
  test_GFunctions()  

