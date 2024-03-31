import numpy as np

def q_ext(r):
    """
    Extinction coefficient parameterization (incl. Mie scattering) based on Mitchell (2000, JAS)
    """

    lmbda = 500.0E-9 # wavelength
    nr = 1.309 # real part of refractive index
    ni = 0.0019 # imaginary part of refractive index

    D = 2.0 * r
    x = 2.0 * np.pi * r / lmbda
    rho = 2.0 * x * ( nr - 1.0 )
    beta = np.arctan( ni / ( nr - 1.0 ) )
    a6 = 1.0

    # Anomalous diffraction approximation (ADA) based on van de Hulst (1981), see (14) in Mitchell (2000)
    Q_ext_ADA = 2.0 - 4.0 * np.exp( -rho * np.tan( beta ) ) * np.cos( beta ) / rho * np.sin( rho - beta ) \
        - 4.0 * np.exp( -rho * np.tan( beta ) ) * ( np.cos( beta ) / rho )**2 * np.cos( rho - 2.0 * beta ) \
            + 4.0 * ( np.cos( beta ) / rho )**2 * np.cos( 2.0 * beta )

    # Tunneling effects
    ra = 0.7393 * nr - 0.6069 # (6)
    rext = ra / 2.0 # (11)

    m = 0.5 # dispersion gamma function, 0.5 has been found to be a good approcimation
    kmean = D / lmbda
    epsilon0 = 0.25 + 0.6 * ( 1.0 - np.exp( -8.0 * np.pi * ni / 3.0 ) )**2 # (8)
    kmax = m / epsilon0

    C3 = rext * ( kmean**m * np.exp( -epsilon0 * kmean ) ) / ( kmax**m * np.exp( -m ) ) # (12)

    # Edge effects
    Q_edge = a6 * ( 1.0 - np.exp( -0.06 * x ) ) * x**(-2.0/3.0) # (17)

    # Final approximation
    Q_ext = ( 1.0 + C3 ) * Q_ext_ADA + Q_edge # (18)
    
    return Q_ext
