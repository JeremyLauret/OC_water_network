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


def signe(a):
    if a < 0:
        return -1
    return 1


def OracleDG(lamda, ind):
    '''
    Compute -Phi(lamda) [criterion of the problem] and G(lamda) [gradient of the criterion].

    ind = 2 : return -Phi(lamda).
    ind = 3 : return -G(lamda).
    ind = 4 : return -Phi(lamda) and -G(lamda).
    '''
    A_0 = (np.dot(np.transpose(Ar), pr) + np.dot(np.transpose(Ad), lamda)) / r
    q_arg = np.array([-signe(A_0[i]) * np.sqrt(np.abs(A_0[i])) for i in range(n)])
    if (ind == 2):
        return (
            - 1 / 3 * np.vdot(q_arg, r * q_arg * np.abs(q_arg)) - np.vdot(pr, np.dot(Ar, q_arg))
            - np.vdot(lamda, np.dot(Ad, q_arg) - fd)
        )
    if (ind == 3):
        return - np.dot(Ad, q_arg) + fd
    if (ind == 4):
        return (
            - 1 / 3 * np.vdot(q_arg, r * q_arg * np.abs(q_arg)) - np.vdot(pr, np.dot(Ar, q_arg))
            - np.vdot(lamda, np.dot(Ad, q_arg) - fd),
            - np.dot(Ad, q_arg) + fd
        )


def OracleDH(lamda, ind):
    '''
    Compute Phi(lamda) [criterion of the problem], G(lamda) [gradient of the criterion]
    and H(lamda) [hessian of the criterion].

    ind = 2 : return Phi(lamda).
    ind = 3 : return G(lamda).
    ind = 4 : return Phi(lamda) and G(lamda).
    ind = 5 : return H(lamda).
    ind = 6 : return G(lamda) and H(lamda).
    ind = 7 : return Phi(lamda), G(lamda) and H(lamda).
    '''
    if ind in [2, 3, 4]:  # Same computations as in OraclePG.
        return OracleDG(lamda, ind)
    A_1 = np.diag(1 / np.sqrt(np.abs(
        r * (np.dot(np.transpose(Ar), pr) + np.dot(np.transpose(Ad), lamda))
    )))
    H = np.dot(Ad, np.dot(A_1, Ad.transpose())) / 2
    if ind == 5:
       return H
    if ind == 6:
       G = OracleDG(lamda, 3)
       return (G, H)
    if ind == 7:
       F, G = OracleDG(lamda, 4)
       return (F, G, H)