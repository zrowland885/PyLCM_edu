import numpy as np
from parameters import *
from micro import *
from tqdm import tqdm

def drop_condensation(particles_list, T_parcel, q_parcel, P_parcel, dt, air_mass_parcel, rho_aero, molecular_weight_aero = 1.):
    dq_liq = 0

    for particle in particles_list:
        dq_liq = dq_liq - particle.M

        e_s = esatw( T_parcel )
        e_a = q_parcel * P_parcel / (q_parcel + r_a / rv)
        supersat = e_a / e_s - 1.0

        thermal_conductivity = 7.94048E-05 * T_parcel + 0.00227011
        diff_coeff = 0.211E-4 * (T_parcel / 273.15) ** 1.94 * (101325.0 / P_parcel)

        G_pre = 1.0 / (rho_liq * rv * T_parcel / (e_s * diff_coeff) + (l_v / (rv * T_parcel) - 1.0) *
                       rho_liq * l_v / (thermal_conductivity * T_parcel))

        r_liq = (particle.M / (particle.A * 4.0 / 3.0 * np.pi * rho_liq)) ** 0.33333333333
        r_N = (particle.Ns / (particle.A * 4.0 / 3.0 * np.pi * rho_aero)) ** 0.33333333333

        afactor = 2.0 * sigma_air_liq(T_parcel) / (rho_liq * rv * T_parcel)
        bfactor = vanthoff_aero * rho_aero * molecular_weight_water / (rho_liq * molecular_weight_aero)

        alpha = 0.036
        r0 = diff_coeff / alpha * np.sqrt(2.0 * np.pi / (rv * T_parcel)) / (1.0 + diff_coeff * l_v ** 2 *
                                                                         e_s / (thermal_conductivity * rv ** 2 *
                                                                               T_parcel ** 3))

        f_vent = 1.0
        D_pre = 0.0
        radiation = 0.0

        r_liq_old = r_liq
        r_liq = radius_liquid_euler_py(r_liq, dt, r0, G_pre, supersat, f_vent, afactor, bfactor, r_N, D_pre, radiation)

        particle.M = particle.A * 4.0 / 3.0 * np.pi * rho_liq * r_liq ** 3
        dq_liq = dq_liq + particle.M

    T_parcel = T_parcel + dq_liq * l_v / cp / air_mass_parcel
    q_parcel = q_parcel - dq_liq / air_mass_parcel

    return particles_list, T_parcel, q_parcel

def esatw(T):
    a = [6.11239921, 0.443987641, 0.142986287e-1,
         0.264847430e-3, 0.302950461e-5, 0.206739458e-7,
         0.640689451e-10, -0.952447341e-13, -0.976195544e-15]

    dT = T - 273.15
    esatw = a[0] + dT * (a[1] + dT * (a[2] + dT * (a[3] + dT * (a[4] + dT * (a[5] + dT * (a[6] + dT * (a[7] + a[8] * dT)))))))
    esatw *= 100.0

    return esatw

def r_equi(S,T,r_aerosol, rho_aero,molecular_weight_aero):
    S_internal = min( S, -0.0001 )
    afactor = 2.0 * sigma_air_liq(T) / ( rho_liq * rv * T ) # curvature effect
    bfactor = vanthoff_aero * rho_aero * molecular_weight_water / ( rho_liq * molecular_weight_aero ) # solute effect

#     Iterative solver ( 0 = S - A / r + B / r^3 => r = ( B / ( A / r - S ) )^(1/3) )
    r_equi_0 = 1.0
    r_equi   = 1.0E-6
    
    while ( abs( ( r_equi - r_equi_0 ) / r_equi_0 ) > 1.0E-20 ):
        r_equi_0 = r_equi
        r_equi   = ( ( bfactor * r_aerosol**3 ) / ( afactor / r_equi - S_internal ) )**(1.0/3.0)
    return(r_equi)

def sigma_air_liq(tabs):
    tabs_c = tabs - 273.15
    #!
    #!--    Pruppacher and Klett (1997), Eq. 5-12
    sigma_air_liq = 75.93 + 0.115 * tabs_c    + 6.818e-2 * tabs_c**2 + 6.511e-3 * tabs_c**3 + 2.933e-4 * tabs_c**4 + 6.283e-6 * tabs_c**5 + 5.285e-8 * tabs_c**6
    sigma_air_liq = sigma_air_liq * 1.0E-3
    
    return(sigma_air_liq)

def radius_liquid_euler(r_ini, dt_int, r0, G_pre, supersat, ventilation_effect, afactor, bfactor, r_aero, D_pre, radiation):
    r_eul = r_ini
    r_eul_old = r_ini

    dt_eul = dt_int
    t_eul = 0.0

    while t_eul < dt_int - 1.0E-20:
        for m in range(500):
            r_eul_3 = r_eul**3
            r_eul_r0 = r_eul + r0

            dr2dt = 2.0 * G_pre * ventilation_effect * (supersat - afactor / r_eul + bfactor * r_aero**3 / r_eul_3 - D_pre * radiation * r_eul) * r_eul / r_eul_r0
            d2r2dtdr2 = G_pre * ventilation_effect * (afactor * r_eul_3 - bfactor * r_aero**3 * (3.0 * r_eul + 2.0 * r0) - r_eul_3 * (D_pre * radiation * r_eul * r_eul_r0 - r0 * supersat)) / (r_eul**4 * r_eul_r0**2)

            dt_eul = min(0.5 * np.abs(1.0 / d2r2dtdr2), dt_int - t_eul, dt_int)

            f = r_eul**2 - r_ini**2 - dt_eul * dr2dt
            dfdr2 = 1.0 - dt_eul * d2r2dtdr2

            r_eul = np.sqrt(np.maximum(r_eul**2 - f / dfdr2, r_aero**2))

            rel_change = np.abs(r_eul - r_eul_old) / r_eul_old
            r_eul_old = r_eul

            if np.all(rel_change < 1.0E-12):
                break

        t_eul += dt_eul
        r_ini = r_eul

    return r_ini

from scipy.optimize import newton

def radius_liquid_euler_py(r_ini, dt_int, r0, G_pre, supersat, ventilation_effect, afactor, bfactor, r_aero, D_pre, radiation):
    def equation(r_eul):
        r_eul_3 = r_eul**3
        r_eul_r0 = r_eul + r0
        dr2dt = 2.0 * G_pre * ventilation_effect * (supersat - afactor / r_eul + bfactor * r_aero**3 / r_eul_3 - D_pre * radiation * r_eul) * r_eul / r_eul_r0
        d2r2dtdr2 = G_pre * ventilation_effect * (afactor * r_eul_3 - bfactor * r_aero**3 * (3.0 * r_eul + 2.0 * r0) - r_eul_3 * (D_pre * radiation * r_eul * r_eul_r0 - r0 * supersat)) / (r_eul**4 * r_eul_r0**2)
        dt_eul = min(0.5 * abs(1.0 / d2r2dtdr2), dt_int)
        f = r_eul**2 - r_ini**2 - dt_eul * dr2dt
        dfdr2 = 1.0 - dt_eul * d2r2dtdr2
        return r_eul**2 - r_ini**2 - dt_eul * dr2dt

    r_eul = newton(equation, r_ini)

    return r_eul