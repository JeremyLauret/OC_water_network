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
    print(np.shape(Bqc))
    q = q0 + Bqc
    if (ind == 2) :
        F = 1 / 3 * np.vdot(q, r * q * np.abs(q)) + np.vdot(pr, np.dot(Ar, q))
        return F
    if (ind == 3):
        G = np.dot(np.transpose(B), r * q * np.abs(q)) + np.dot(np.transpose(B), np.dot(np.transpose(Ar), pr))
        return G
    if (ind == 4):
        F = 1 / 3 * np.vdot(q, r * q * np.abs(q)) + np.vdot(pr, np.dot(Ar, q))
        G = np.dot(np.transpose(B), r * q * np.abs(q)) + np.dot(np.transpose(B), np.dot(np.transpose(Ar), pr))
        return (F, G)

    
def OraclePH(qc, ind):
    if (ind == 2) :
        F = 0
        Bqc = np.dot(B,qc)
        F = (1/3)*np.vdot(q0 + Bqc, (r*(q0 + Bqc)*np.abs(q0 + Bqc))) + np.vdot(pr, np.dot(Ar,q0 + Bqc))
        return(F)
    if (ind == 3):
        G = np.zeros(n - md)
        Bqc = np.dot(B,qc)
        G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
        return (G)
    if (ind == 4):
        F = 0
        G = np.zeros(n - md)
        Bqc = np.dot(B,qc)
        F = (1/3)*np.vdot(q0 + Bqc, (r*(q0 + Bqc)*np.abs(q0 + Bqc))) + np.vdot(pr, np.dot(Ar,q0 + Bqc))
        G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
        return(F,G)
    if (ind == 5):
        RQ = np.eye(n)
        for i in range(n):
            RQ[i][i] = r[i]*np.abs(q0)[i]
        H = np.dot(np.dot(np.transpose(B),RQ),B)
        return(H)
    if (ind == 6):
        G = np.zeros(n - md)
        Bqc = np.dot(B,qc)
        G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
        RQ = np.eye(n)
        for i in range(n):
            RQ[i][i] = r[i]*np.abs(q0)[i]
        H = np.dot(np.dot(np.transpose(B),RQ),B)
        return(G,H)
    if (ind == 7):
        F = 0
        G = np.zeros(n - md)
        Bqc = np.dot(B,qc)
        F = (1/3)*np.vdot(q0 + Bqc, (r*(q0 + Bqc)*np.abs(q0 + Bqc))) + np.vdot(pr, np.dot(Ar,q0 + Bqc))
        G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
        RQ = np.eye(n)
        for i in range(n):
            RQ[i][i] = r[i]*np.abs(q0 + Bqc)[i]
        #print(RQ)
        H = 2*np.dot(np.dot(np.transpose(B),RQ),B)
        return(F,G,H)
    