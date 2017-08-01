import numpy as np

class canopyLayer(object):

  def __init__(self):
    self.leaf_r=0.1
    self.leaf_t=0.1
    self.lai=2.0    
    self.G=0.5


class mwCanopy(object):

  def __init__(self, nLayers):
  
    self.mu=1.0
    self.soil_r=0.1
    self.propDif=0.0
    
    #set up canopy layers
    self.layers=[]
    self.nLayers=nLayers
    for i in xrange(self.nLayers):
      self.layers.append(canopyLayer())

  def B_direct(self,layer):
    """Beta for spherical LAD from Sellers
    """

    G=self.layers[layer].G    
    leaf_t=self.layers[layer].leaf_t
    leaf_r=self.layers[layer].leaf_r
    
    w=leaf_r+leaf_t    
    ssa=0.5*w*(1.-self.mu*np.log((self.mu+1.)/self.mu))
    K=G/self.mu
    muBar=1.
    
    return (1./w)*ssa*(1.+muBar*K)/(muBar*K)


  def B_diffuse(self, layer):
    """Beta_diffuse for spherical LAD from Sellers
    """
    
    leaf_t=self.layers[layer].leaf_t
    leaf_r=self.layers[layer].leaf_r

    w=leaf_r+leaf_t
    d=leaf_r-leaf_t
    sqcost=1./3.
    
    return 0.5*(w+d*sqcost)/w


  def getGamma(self,layer):
    """Get the coefficients for the Meador and Weaver
    two stream model that are consistent with the 
    Sellers model.
    """
    
    leaf_t=self.layers[layer].leaf_t
    leaf_r=self.layers[layer].leaf_r

    w=leaf_r+leaf_t    
    B=self.B_diffuse(layer)
    B0=self.B_direct(layer)
    
    self.layers[layer].g1=2.-(1.-B)*w*2.
    self.layers[layer].g2=2.*w*B
    self.layers[layer].g3=B0
    self.layers[layer].g4=1.-self.layers[layer].g3
    

  def rtVegLayerDir(self,layer):
    """Get reflectance and transmittance for
    a single canopy layer under direct illumination
    """
    
    self.getGamma(layer)

    G=self.layers[layer].G
    lai=self.layers[layer].lai
    leaf_t=self.layers[layer].leaf_t
    leaf_r=self.layers[layer].leaf_r

    g1=self.layers[layer].g1
    g2=self.layers[layer].g2
    g3=self.layers[layer].g3
    g4=self.layers[layer].g4

    w=leaf_r+leaf_t    
    tau=lai*G
    
    a1=g1*g4+g2*g3
    a2=g1*g3+g2*g4
    k=np.sqrt(g1*g1-g2*g2)
  
    D=(1.-k*k*self.mu*self.mu)*((k+g1)*np.exp(k*tau)+(k-g1)*np.exp(-k*tau))

    #reflectance:
    F=(1.-k*self.mu)*(a2+k*g3)*np.exp(k*tau)
    G=(1.+k*self.mu)*(a2-k*g3)*np.exp(-k*tau)
    H=2.*k*(g3-a2*self.mu)*np.exp(-tau/self.mu)
      
    R=w/D*(F-G-H)
    
    #transmittance
    F=(1.+k*self.mu)*(a1+k*g4)*np.exp(k*tau)
    G=(1.-k*self.mu)*(a1-k*g4)*np.exp(-k*tau)
    H=2.*k*(g4+a1*self.mu)*np.exp(tau/self.mu)

    T=np.exp(-tau/self.mu)*(1.-w/D*(F-G-H))
    
    return R,T
  
  
  def tUncollidedLayerDir(self,layer):
    """Uncollided transmission of collimated radiation
    """
    G=self.layers[layer].G
    lai=self.layers[layer].lai
    return np.exp(-G*lai/self.mu)
    
    
  def rtVegLayerDif(self,layer):
    """Get reflectance and transmittance for
    a single canopy layer under diffuse illumination
    """
  
    self.getGamma(layer)

    G=self.layers[layer].G
    lai=self.layers[layer].lai

    g1=self.layers[layer].g1
    g2=self.layers[layer].g2
    
    tau=lai*G    
    k=np.sqrt(g1*g1-g2*g2)
    
    D=k+g1+(k-g1)*np.exp(-2.*k*tau)
    
    R=(g2*(1.-np.exp(-2.*k*tau)))/D
    T=(2.*k*np.exp(-k*tau))/D
    
    return R, T
    

  def getFluxesCanopyDif(self):
    """Calculate the fluxes between each layer for the 
    whole canopy under DIFFUSE illumination
    """
    
    #work from lowest layer up using adding to
    #calculate values of reflectance and transmission        
    R_last=self.soil_r
    for layer in xrange(self.nLayers-1,-1,-1):
      R,T=self.rtVegLayerDif(layer)
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
        Iup_below=self.layers[layer].Idn_dif*self.soil_r
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
    Ru_last=self.soil_r
    Rc_last=self.soil_r
    for layer in xrange(self.nLayers-1,-1,-1):

      Ru,Tu=self.rtVegLayerDir(layer)
      Rc,Tc=self.rtVegLayerDif(layer)
      U=self.tUncollidedLayerDir(layer)
      
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
      U=self.tUncollidedLayerDir(layer)
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
        Iup_below=self.layers[layer].Idn_dir*self.soil_r
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

  from sellersTwoStream import twoStream

  nLayers=10
  c=mwCanopy(nLayers)
  c.mu=0.6
  c.soil_r=0.01
  c.propDif=0.5
  c.getFluxes()
    
  t=twoStream()
  t.nLayers=nLayers
  t.lai=c.layers[0].lai*nLayers
  t.propDif=c.propDif
  t.mu=c.mu
  t.leaf_r=c.layers[0].leaf_r
  t.leaf_t=c.layers[0].leaf_t
  t.soil_r=c.soil_r
  #t.userLayerLAIMap=False
  Iup, Idn, Iab, Iab_dLai = t.getFluxes()
  
  space="    "
  for L in xrange(nLayers):
    print "Iup[%d]"%L, c.layers[L].Iup, Iup[L],   space, c.layers[L].Iup-Iup[L]
    print "Idn[%d]"%L, c.layers[L].Idn, Idn[L+1], space, c.layers[L].Idn-Idn[L+1]
    print "Iab[%d]"%L, c.layers[L].Iab, Iab[L+1], space, c.layers[L].Iab-Iab[L+1]
  

  

