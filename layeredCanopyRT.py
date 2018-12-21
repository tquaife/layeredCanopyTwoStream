""" layeredCanopyRT.py

A multi-layered two-stream scheme for vegetation with vertically 
varying properties. Use the Meador and Weavor (1980) two-stream 
equations to calculate the optical properties of individual layers 
and then the formalism of adding to combine them. Ifthe canopy is 
vertically homogeneous the results are numerically identical to 
the Sellers' two-stream model.
    
Copyright (C) 2017 Tristan Quaife

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
from leafGeometry import leafGeometry

#import warnings
#warnings.filterwarnings("error")

class canopyStructure( ):

  def __init__(self):
    """Various description of the modulation of the
    optical depth as a function of ray geometry and
    canopy structure.
    """
    
    self.pinty_a=1.0
    self.pinty_b=0.0

  def zeta_noStruct(self,mu):
    return 1.0
    
  def zeta_pinty(self,mu):
    return self.pinty_a+self.pinty_b*(1-mu)


class canopyLayer(leafGeometry,canopyStructure):

  def __init__(self,mu):
    """Define the optical properties for a single canopy
    layer using the Meador and Weaver two stream equations
    
    Includes a number of options for calculating the upscatter
    parameters.
    """
    
    leafGeometry.__init__(self)
    canopyStructure.__init__(self)

    self.mu=mu
    self.leaf_r=0.1
    self.leaf_t=0.1
    self.lai=.2
    
    self.gamma_scaling_diffuse="delta"    
    self.gamma_scaling_direct="delta"    
    
    self.setupJULES()


  @property
  def w(self):
    return self.leaf_r+self.leaf_t  

  @property
  def d(self):
    return self.leaf_r-self.leaf_t  
  
  

  def setupJULES(self):
    """
    Wire-up the methods so that the flux
    is calculated as in the JULES code
    """

    self.Z=self.zeta_noStruct
    self.G=self.G_JULES
    self.K=self.K_JULES
    self.muBar=self.muBar_JULES
    self.B_direct=self.B_direct_JULES  
    self.B_diffuse=self.B_diffuse_JULES  


  def setupCLM(self, chiL):
    """
    Wire-up the methods so that the flux
    is calculated as in the CLM code
    """
    self.CLM_chiL=chiL
    self.Z=self.zeta_noStruct
    self.G=self.G_CLM
    self.K=self.K_CLM
    self.muBar=self.muBar_CLM
    self.B_direct=self.B_direct_CLM  
    self.B_diffuse=self.B_diffuse_CLM  


      
  # ========================================
  # K methods ==============================
  # ========================================

  def __K(self):
    """
    Private method containing the common parts
    of the calculation of K
    """
    return self.G(self.mu)/self.mu

  def K_generic(self):
    """
    Optical depth per unit leaf area in the
    direction mu calculated for an arbitrary 
    G/GZ function 
    """
    return self.__K()*self.Z(self.mu)

  def K_JULES(self):
    """
    Optical depth per unit leaf area in the
    direction mu as calculated in JULES
    """
    return self.__K()
  
  def K_CLM(self):
    """
    Optical depth per unit leaf area in the
    direction mu as calculated in CLM
    """
    return self.__K()
  

  # ========================================
  # muBar methods  =========================
  # ========================================
  
  def muBar_uniformG_pintyZ_analytic(self):
    """
    Analytical solution for the average inverse 
    diffuse optical depth per unit leaf area for
    a uniform leaf angle distribution and the Pinty
    Zeta function. Only valid for a>0 and b>=0. 
    """
    a=self.pinty_a
    b=self.pinty_b
    if b < 0.00001:
      return 1/a
    return -2/(b*b) * ((a+b)*np.log(a/(a+b))+b)



  def muBar_generic(self):
    """
    Average inverse diffuse optical depth per unit leaf area
    calculated for an arbitrary G/GZ function 
    """
    out=integrate.quadrature( self.__muBar_generic_integ,0,1,vec_func=False)
    return out[0] 
    
  def __muBar_generic_integ(self,muDash):
    """
    Private method to be integrated to find muBar
    """
    return muDash/(self.G(muDash)*self.Z(muDash))
  

  def muBar_CLM(self):  
    """
    Average inverse diffuse optical depth per unit leaf area
    as calculated in the CLM implementation 
    """
  
    p1=self.CLM_phi1()
    p2=self.CLM_phi2()
    
    return 1./p2*(1.-p1/p2*np.log((p1+p2)/p1))


  def muBar_JULES(self):
    """
    Average inverse diffuse optical depth per unit leaf area
    as calculated in the JULES implementation 
    
    n.b. is unity in both cases
    """
    
    if self.JULES_lad=='uniform':
      return 1.0
    elif self.JULES_lad=='horizontal':
      return 1.0
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad


  # ========================================
  # Volume single scattering albedo ========
  # ========================================


  def volssa_uniformG_pintyZ_analytic(self):
    """
    Analytical solution for the single scattering 
    albedo for a uniform leaf angle distribution and the Pinty
    Zeta function. Only valid for a>0 and b>=0. 
    """

    mu=self.mu
    a=self.pinty_a
    b=self.pinty_b
    w=self.leaf_r+self.leaf_t
        
    f=self.zeta_pinty(mu)
    g=f-mu*b
    h=mu*a+mu*b
        
    return w/2.*f*(h*np.log(h/(g+h))+g)/(g*g)    
    

  def volssa_generic(self):
    """
    The volume single scattering albedo for any
    G and Zeta function
    
    See eqn 5 & 7 of Sellers 1985
    """
    out=integrate.quadrature( self.__volssa_generic_integ,0,1,vec_func=False)
    return out[0]*(self.leaf_r+self.leaf_t)*0.5

    
  def __volssa_generic_integ(self,muDash):
    """
    Private method which is integrated to find single scattering albedo
    """
    mu=self.mu
    t1=muDash*(self.G(mu)*self.Z(mu))
    t2=mu*(self.G(muDash)*self.Z(muDash))+muDash*(self.G(mu)*self.Z(mu))

    return t1/t2


  def volssa_CLM(self):
    """
    The volume single scattering albedo as defined
    in CLM. Eqn 3.15 in TechNote 4.
    """
    
    w=self.leaf_r+self.leaf_t
    G=self.G(self.mu)
    
    p1=self.CLM_phi1( )
    p2=self.CLM_phi2( )
    
    t1=w/2.
    t2=G/(self.mu*p2+G)
    t3=self.mu*p1/(self.mu*p2+G)
    t4=np.log((self.mu*p1+self.mu*p2+G)/(self.mu*p1))

    return t1*t2*(1.-t3*t4)


  def volssa_JULES(self):
    """
    The volume single scattering albedo as defined
    in JULES.     
    """

    w=self.leaf_r+self.leaf_t

    if self.JULES_lad=='uniform':
      return 0.5*w*(1.-self.mu*np.log((self.mu+1.)/self.mu))
    elif self.JULES_lad=='horizontal':
      return w/4.
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad




  # ========================================
  # Direct upscatter methods ===============
  # ========================================

  def B_direct_CLM(self):
    """
    Direct upscatter parameter as defined in CLM
    """
    ssa=self.volssa_CLM()
    return self.B_direct_Dickinson(ssa)


  def B_direct_JULES(self):
    """
    Direct upscatter parameter as defined in JULES
    """
    ssa=self.volssa_JULES()
    return self.B_direct_Dickinson(ssa)


  def B_direct_Dickinson_generic_ssa(self):
    """
    Direct upscatter parameter as defined by Dickinson
    but using a generic formulation for the SSA
    """
    ssa=self.volssa_generic()
    return self.B_direct_Dickinson(ssa)


  def B_direct_Dickinson_volssa_uniformG_pintyZ_analytic(self):
    """
    Direct upscatter parameter as defined by Dickinson
    but using a generic formulation for the SSA
    """
    ssa=self.volssa_uniformG_pintyZ_analytic()
    return self.B_direct_Dickinson(ssa)



  def B_direct_Dickinson(self,ssa):
    """
    Direct upscatter parameter as defined by Dickinson
    both JULES and CLM use this formulation but differ
    in the calculation of the single scattering albedo.
    """

    w=self.leaf_r+self.leaf_t
    return (1./w)*ssa*(1.+self.muBar()*self.K())/(self.muBar()*self.K())

  
  def B_direct_generic(self):
    """
    Compute the direct upscatter according to Pinty et al. 2006
    (eqn A3)
    """
    w=self.leaf_r+self.leaf_t  
    d=self.leaf_r-self.leaf_t  
    intg=self.integ_cosSq_gDash()

    return (0.5/w)*(w+d*self.mu/(self.G(self.mu)*self.Z(self.mu))*intg)
    
    
  def integ_cosSq_gDash(self):
    """
    Integrate cos^2(theta)*gDash(theta) for calculating 
    upscatter parameters. See Pinty et al. 2006.
    """
    intg=integrate.quad(self.cos2_gDash,0,np.pi/2.)      
    return intg[0]


  def cos2_gDash(self, theta):  
    """ 
    Method to be integrated to find cos^2(theta)*gDash(theta) 
    """
    mu=np.cos(theta)
    return self.gDash(mu)*mu**2



  # ========================================
  # Diffuse upscatter methods ==============
  # ========================================


  def B_diffuse_CLM(self):
    """
    The Diffuse upscatter as calculated in CLM
    as a function of chiL
    """    
    w=self.leaf_r+self.leaf_t
    d=self.leaf_r-self.leaf_t
    
    return (0.5*(w+d*((1.+self.CLM_chiL)/2.)**2.))/w


  def B_diffuse_JULES(self):
    """
    The Diffuse upscatter as calculated in JULES
    """    
    w=self.leaf_r+self.leaf_t
    d=self.leaf_r-self.leaf_t
    if self.JULES_lad=='uniform':
      sqcost=1./3.
    elif self.JULES_lad=='horizontal':
      sqcost=1.0
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad
    
    return 0.5*(w+d*sqcost)/w


  def B_diffuse_generic(self):
    """
    Compute the diffuse upscatter according to Pinty et al. 2006
    """
    w=self.leaf_r+self.leaf_t  
    d=self.leaf_r-self.leaf_t  
    intg=self.integ_cosSq_gDash()

    return (0.5/w)*(w+d*intg)


  # ========================================
  # Gamma coefficients and the Meador ======
  # and Weaver two-stream soultions ========
  # ========================================


  def getGamma(self,method="delta"):
    """Get the coefficients for the Meador and Weaver
    two stream model that are consistent with the 
    Sellers model.
    """
    
    B=self.B_diffuse()
    B0=self.B_direct()
    
    if method=="delta":    
      scale=1./(self.G(self.mu)*self.muBar())
    elif method=="quad": 
      #modified quadrature
      scale=np.sqrt(3.)
    else:
      raise TypeError("Unknown gamma scaling type:%s",method)
      
    
    g1=(1.-(1.-B)*self.w) *scale
    g2=(self.w*B) * scale
         
    g3=B0
    g4=1.-g3
    
    return g1,g2,g3,g4

  def rtDirect(self):
    """Get reflectance and transmittance for
    a single canopy layer under direct illumination
    """
    
    g1,g2,g3,g4=self.getGamma(method=self.gamma_scaling_direct)

    tau=self.lai*self.G(self.mu)*self.Z(self.mu)
    tauD=tau
    
    a1=g1*g4+g2*g3
    a2=g1*g3+g2*g4
    k=np.sqrt(g1*g1-g2*g2)
  
    D=(1.-k*k*self.mu*self.mu)*((k+g1)*np.exp(k*tauD)+(k-g1)*np.exp(-k*tauD))

    #reflectance:
    F=(1.-k*self.mu)*(a2+k*g3)*np.exp(k*tauD)
    G=(1.+k*self.mu)*(a2-k*g3)*np.exp(-k*tauD)
    H=2.*k*(g3-a2*self.mu)*np.exp(-tau/self.mu)
      
    R=self.w/D*(F-G-H)
    
    #transmittance
    F=(1.+k*self.mu)*(a1+k*g4)*np.exp(k*tauD)
    G=(1.-k*self.mu)*(a1-k*g4)*np.exp(-k*tauD)
    H=2.*k*(g4+a1*self.mu)*np.exp(tau/self.mu)
    T=np.exp(-tau/self.mu)*(1.-self.w/D*(F-G-H))
    
    return R,T
  
  
  def tUncollidedDirect(self):
    """Uncollided transmission of collimated radiation
    """
    return np.exp(-self.G(self.mu)*self.Z(self.mu)*self.lai/self.mu)
    

  def rtDiffuse(self):
    """Get reflectance and transmittance for
    a single canopy layer under diffuse illumination
    """
  
    g1,g2,g3,g4=self.getGamma(method=self.gamma_scaling_diffuse)
    
    tau=self.lai*self.G(self.mu)*self.Z(self.mu)    
    #tau=self.lai/self.muBar_uniformG_pintyZ_analytic()
    
    k=np.sqrt(g1*g1-g2*g2)
    
    D=k+g1+(k-g1)*np.exp(-2.*k*tau)
    R=(g2*(1.-np.exp(-2.*k*tau)))/D
    T=(2.*k*np.exp(-k*tau))/D
    
    return R, T



class layeredCanopy(object):

  def __init__(self, nLayers, mu):
    """Calculates vertical profile of canopy fluxes of
    a series of layers each with their own optical 
    properties using adding.
    """
    
    self.mu=mu
    self.lower_boundary_r=0.1
    self.propDif=0.0
    
    #set up canopy layers
    self.layers=[]
    self.nLayers=nLayers
    for i in xrange(self.nLayers):
      self.layers.append(canopyLayer(mu))


  def getFluxesCanopyDif(self):
    """Calculate the fluxes between each layer for the 
    whole canopy under DIFFUSE illumination
    """
    
    #work from lowest layer up using adding to
    #calculate values of reflectance and transmission        
    R_last=self.lower_boundary_r
    for layer in xrange(self.nLayers-1,-1,-1):
      R,T=self.layers[layer].rtDiffuse()
      Z=R_last*R  
      R=R+T*T*R_last*(1.+Z/(1.-Z))
      T=T*(1.+Z/(1.-Z))
      R_last=R
      
      self.layers[layer].Iup_dif=R_last
      self.layers[layer].Idn_dif=T
    
    #work back down through the layers to get
    #fluxes by normalising by total transmission
    T=1.
    for layer in xrange(self.nLayers):
      t=self.layers[layer].Idn_dif
      self.layers[layer].Iup_dif*=T
      self.layers[layer].Idn_dif*=T
      T*=t
      
    #calculate absorption by solving energy balance
    for layer in xrange(self.nLayers):
      if layer==0:
        Idn_above=1
      else:
        Idn_above=self.layers[layer-1].Idn_dif
      if layer==(self.nLayers-1):
        Iup_below=self.layers[layer].Idn_dif*self.lower_boundary_r
      else:
        Iup_below=self.layers[layer+1].Iup_dif
      
      Idn=self.layers[layer].Idn_dif
      Iup=self.layers[layer].Iup_dif
      self.layers[layer].Iab_dif=np.abs(Iup_below-Iup+Idn_above-Idn)
      

  def getFluxesCanopyDir(self):
    """Calculate the fluxes between each layer for the 
    whole canopy under DIRECT illumination
    
    n.b. 
    
    Ru = hemispheric reflectance arising from uncollided beam 
    Rc = hemispheric reflectance arising from collided 
    etc... 
    """
    
    #work from lowest layer up using adding to
    #calculate values of reflectance and transmission        
    Ru_last=self.lower_boundary_r
    Rc_last=self.lower_boundary_r
    for layer in xrange(self.nLayers-1,-1,-1):

      Ru,Tu=self.layers[layer].rtDirect()
      Rc,Tc=self.layers[layer].rtDiffuse()
      U=self.layers[layer].tUncollidedDirect()
      
      Z=Rc_last*Rc  
    
      #Collimated beam
      RuNew=Ru+U*Ru_last*Tc*(1.+Z/(1.-Z)) + (Tu-U)*Rc_last*Tc*(1.+Z/(1.-Z))
      TuNew=U+U*Ru_last*Rc*(1.+Z/(1.-Z)) + (Tu-U)*(1.+Z/(1.-Z))
      
      #Collided radiation
      RcNew=Rc+Tc*Rc_last*Tc*(1.+Z/(1.-Z))
      TcNew=Tc*(1.+Z/(1.-Z))
      
      Ru_last=RuNew
      Rc_last=RcNew

      self.layers[layer].Iup_dir_u=RuNew
      self.layers[layer].Iup_dir_c=RcNew
      self.layers[layer].Idn_dir_u=TuNew
      self.layers[layer].Idn_dir_c=TcNew


    #work back down through the layers to get
    #fluxes by normalising by total transmission
    Tu=1.
    Tc=0.
    for layer in xrange(self.nLayers):
            
      self.layers[layer].Iup_dir_u*=Tu
      self.layers[layer].Idn_dir_u*=Tu
      self.layers[layer].Iup_dir_c*=Tc
      self.layers[layer].Idn_dir_c*=Tc
      self.layers[layer].Iup_dir=self.layers[layer].Iup_dir_c+self.layers[layer].Iup_dir_u
      self.layers[layer].Idn_dir=self.layers[layer].Idn_dir_c+self.layers[layer].Idn_dir_u
      
      #Keep track of how much uncollided
      #radiation there is incident on next
      #layer down:
      U=self.layers[layer].tUncollidedDirect()
      Tu*=U
      #downward source term (note that Idn_dir_u 
      #has already been normalised by the incoming
      #collimated beam).
      S=self.layers[layer].Idn_dir_u-Tu           
      #Total collided radiation (again, note
      #that Idn_dir_c is normalised already):
      Tc=self.layers[layer].Idn_dir_c+S
               
    #calculate absorption by solving energy balance
    for layer in xrange(self.nLayers):
      if layer==0:
        Idn_above=1
      else:
        Idn_above=self.layers[layer-1].Idn_dir
      if layer==(self.nLayers-1):
        Iup_below=self.layers[layer].Idn_dir*self.lower_boundary_r
      else:
        Iup_below=self.layers[layer+1].Iup_dir
      
      Idn=self.layers[layer].Idn_dir
      Iup=self.layers[layer].Iup_dir
      self.layers[layer].Iab_dir=np.abs(Iup_below-Iup+Idn_above-Idn)
    
  def getFluxes(self):
    p=self.propDif
    self.getFluxesCanopyDir()
    self.getFluxesCanopyDif()
    for layer in xrange(self.nLayers):
      self.layers[layer].Iup=(1.-p)*self.layers[layer].Iup_dir+p*self.layers[layer].Iup_dif
      self.layers[layer].Idn=(1.-p)*self.layers[layer].Idn_dir+p*self.layers[layer].Idn_dif
      self.layers[layer].Iab=(1.-p)*self.layers[layer].Iab_dir+p*self.layers[layer].Iab_dif
    
        
    
if __name__=="__main__":
    pass
  

  

