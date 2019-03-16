# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:15:14 2019

@author: dharc
"""


import numpy as np
from Structures_N import *


def OraclePG(qc, ind):
    '''
    Compute F(qc) [criterion of the problem] and G(qc) [gradient of the criterion].

    ind = 2 : return F(qc).
    ind = 3 : return G(qc).
    ind = 4 : return F(qc) and G(qc).
    '''
    Bqc = np.dot(B, qc)
    q = q0 + Bqc
    if ind == 2 :
        F = 1 / 3 * np.vdot(q, r * q * np.abs(q)) + np.vdot(pr, np.dot(Ar, q))
        return F
    if ind == 3:
        G = np.dot(np.transpose(B), r * q * np.abs(q)) + np.dot(np.transpose(B), np.dot(np.transpose(Ar), pr))
        return G
    if ind == 4:
        F = 1 / 3 * np.vdot(q, r * q * np.abs(q)) + np.vdot(pr, np.dot(Ar, q))
        G = np.dot(np.transpose(B), r * q * np.abs(q)) + np.dot(np.transpose(B), np.dot(np.transpose(Ar), pr))
        return (F, G)

    
def OraclePH(qc, ind):
    '''
    Compute F(qc) [criterion of the problem], G(qc) [gradient of the criterion]
    and H(qc) [hessian of the criterion].

    ind = 2 : return F(qc).
    ind = 3 : return G(qc).
    ind = 4 : return F(qc) and G(qc).
    ind = 5 : return H(qc).
    ind = 6 : return G(qc) and H(qc).
    ind = 7 : return F(qc), G(qc) and H(qc).
    '''
    if ind in [2, 3, 4]: # Same computations as in OraclePG.
        return OraclePG(qc, ind)
    Bqc = np.dot(B, qc)
    q = q0 + Bqc
    H = 2 * np.dot(np.dot(np.transpose(B), np.diag(r * np.abs(q))), B)
    if ind == 5:
        return H
    if ind == 6:
        G = OraclePG(qc, 3)
        return (G, H)
    if ind == 7:
        F, G = OraclePG(qc, 4)
        return (F, G, H)
    