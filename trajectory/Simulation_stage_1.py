# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:31:11 2018

@author: lbrevaul
"""

import numpy as np

import trajectory.Command_stage_1 as Cmd
import constants as Cst
import specifications as Spec
import geopy.distance as dist
from trajectory.modele_atmos import compute_atmos

def simulation_stage_1(t,x, Parameters):


    r = x[0] 
    V = x[1]
    gamma = x[2]
    longi = x[3]
    m = x[4]

    #Parameters
    Diameter = Parameters['geometry']['Diameter']
    Exit_nozzle_area = Parameters['geometry']['Exit_nozzle_area']
    Mass_f = Parameters['masses']['Mass_f']
    Interp_CX_stage_1 = Parameters['aero']['Interp_CX_stage_1']
    Isp = Parameters['propulsion']['Isp']
    Mass_flow_rate_0 = Parameters['propulsion']['Mass_flow_rate']
    N_eng = Parameters['propulsion']['N_eng']
    Pitch_over_duration = Parameters['command']['Pitch_over_duration']
    Delta_theta_pitch_over = Parameters['command']['Delta_theta_pitch_over']
    Delta_vertical_phase = Parameters['command']['Delta_vertical_phase']
    interp_theta_stage_1 = Parameters['command']['Interp_theta_stage_1']
    Integration = Parameters['simu']['Mode_simu']


    #Launcher characteristics for drag calculation
    S_ref = np.pi * Diameter**2/4.
    
    #Compute current atmosphere
    alt = r-Cst.RT
    (Temp,Pa,rho,c) = compute_atmos(alt)
    Mach = V/c
    g_current = Cst.mu/((alt+Cst.RT)**2)


    #Control command to get the command associated to the vehicle state 
    cmd_tmp= Cmd.Command_stage_1(t,m,N_eng,Pa,Exit_nozzle_area,Pitch_over_duration,\
                                                Delta_theta_pitch_over,Mass_f,Delta_vertical_phase,\
    											gamma,rho,V,Mass_flow_rate_0,Isp,alt,\
                                                interp_theta_stage_1)   
    
    
    
    thrust= cmd_tmp[0] #thrust
    alpha= cmd_tmp[1]  #angle of attack
    theta= cmd_tmp[2]  #pitch angle
    Mass_flow_rate= cmd_tmp[3] #engines mass flow rate

    #Aerodynamic forces
    CX = Interp_CX_stage_1(Mach,np.abs(alpha)*180/np.pi)  #drag coefficient as a function of Mach and angle of attack

    #Equations of motion    
    r_dot = V* np.sin(gamma) #gradient of radius
    V_dot = -0.5*rho*S_ref*CX*V**2./m - g_current*np.sin(gamma) + thrust*np.cos(theta-gamma)/m #gradient of velocity
    gamma_dot = (V/r-g_current/V)*np.cos(gamma) + thrust*np.sin(theta-gamma)/(m*V)  #gradient of flight path angle
    longi_dot = V*np.cos(gamma)/r #gradient of longitude
    m_dot = - Mass_flow_rate #gradient of vehicle mass

    dx = np.zeros([1,5])
    dx[0][0] = r_dot
    dx[0][1] = V_dot
    dx[0][2] = gamma_dot
    dx[0][3] = longi_dot
    dx[0][4] = m_dot

    
    if alt<0. :
        dx = np.zeros([1,5])
    
    if Integration == 1.:
         #during integration just return the current state
        return dx[0]
    
    else:
        #during post traitment return all the required data to be saved
        #Load factors to save data: Pdyn, Flux, Distance    
        Pdyn = 0.5*rho*V**2
        flux= Pdyn*V
        lat = Spec.specifications['launch_site']['latitude']
        
        distance = dist.great_circle((Spec.specifications['launch_site']['latitude'],Spec.specifications['launch_site']['longitude']),\
                                     (lat,longi*180/np.pi+Spec.specifications['launch_site']['longitude'])).km
     
        CA = CX*np.cos(np.abs(alpha))  #aerodynamical force
        NX = (thrust-Pdyn*CA*S_ref)/(m*Cst.mu/((alt+Cst.RT)**2))  #axial load factor

        return (r, V, gamma, longi, m, NX,
                Mach,Pdyn,flux,
                alt,alpha,theta,
                rho, CX,thrust,Mass_flow_rate,distance,lat)