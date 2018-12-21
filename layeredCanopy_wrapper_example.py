#!/usr/bin/python
import numpy as np
from layeredCanopyRT import layeredCanopy as lrt

def check_is_list_or_nparray(var):
    """Check that a variable is either a python list or a numpy array.
    If it is not then raise a TypeError.
    If it is a list convert it to a numpy array.
    Return the numpy array.
    """
    
    if isinstance(var,list):
        var=np.array(var)
    if not isinstance(var,np.ndarray):
        raise TypeError("variable must be a list or a numpy array")
    return var


def get_layered_canopy_fluxes(lai, leaf_r, leaf_t, soil_r, mu, prop_diffuse=0.0, chiL=None):
    """Calculate the albedo, total transmission for a canopy and 
    corresponding absorption (i.e. fapar in the par region) per layer.
    The number of layers in the canopy is defined by the length of the
    arrays for the lai and leaf properties. If chiL is not defined then
    the code sets up each layer with a uniform leaf angle distribution.

    The definition of the normalised fluxes is such that:    
    albedo+absrpt.sum()+(1.0-soil_r)*transm == 1
        
    Arguments:
    
    lai    -- list/numpy array of lai values for each layer
    leaf_r -- list/numpy array of leaf reflectance values for each layer
    leaf_t -- list/numpy array of leaf transmittance values for each layer
    soil_r -- scalar value for the lower boundary (soil) reflectance 
    mu     -- scalar value for the cosine of the solar zenith angle
    chiL   -- a parameter that defines the leaf angle distribution
    
    Returns:
    
    albedo -- scalar - the proportion of incident irradiance that is reflected
    transm -- scalar - total transmittance at the bottom of canopy
    absrpt -- numpy array containing the proportion absorbed top-of-canopy 
              irradiance in each layer. The is layer fapar if the values leaf_r 
              and leaf_t are representative of the PAR domain. The sum of the
              array is then equivalent to total canopy fAPAR
    
    
    """

    #check arguments are the correct data types:
    lai=check_is_list_or_nparray(lai)
    leaf_r=check_is_list_or_nparray(leaf_r)
    leaf_t=check_is_list_or_nparray(leaf_t)
    if chiL is not None:
        chiL=check_is_list_or_nparray(chiL)

    #get the length of the input arrays and check:
    #they are the same:
    n_layers=len(lai)
    if n_layers != len(leaf_r):
        raise Exception("input arrays must be the same length")
    if n_layers != len(leaf_t):
        raise Exception("input arrays must be the same length")
    if chiL is not None:
        if n_layers != len(chiL):
            raise Exception("input arrays must be the same length")

    #initialise the canopy:
    canopy=lrt(n_layers,mu)
    canopy.lower_boundary=soil_r
    canopy.propDif=prop_diffuse

    #set properties and methods for individual layers:
    for i in xrange(n_layers):
        canopy.layers[i].lai=lai[i]
        canopy.layers[i].leaf_r=leaf_r[i]
        canopy.layers[i].leaf_t=leaf_t[i]        
        if chiL is not None:
            canopy.layers[i].setupCLM(chiL[i])

                
    #calculate normalised fluxes:
    canopy.getFluxes()
    
    #retrieve flux variables
    albedo=canopy.layers[0].Iup
    transm=canopy.layers[n_layers-1].Idn    
    absrpt=np.zeros(np.shape(lai))
    for i in xrange(n_layers):
        absrpt[i]=canopy.layers[i].Iab    

    #return the results
    return albedo, transm, absrpt
    
   

        
if __name__=="__main__":

    lai=np.array([1.0,1.0,1.0])
    leaf_r=np.array([0.20,0.15,0.10])    
    leaf_t=np.array([0.18,0.11,0.08])
    chiL=np.array([0.001,0.001,0.001])
    chiL=None
    soil_r=0.1
    mu=1.0
    prop_diff=0.0
        
    r,t,a = get_layered_canopy_fluxes(lai,leaf_r,leaf_t,soil_r,mu,prop_diff,chiL=chiL)
    
    print r,t,a
    print r+a.sum()+(1.0-soil_r)*t
    
