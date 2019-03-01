# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:15:14 2019

@author: dharc
"""


import numpy as np
import random
from numpy.linalg import inv
from numpy import dot

from Probleme_R import *
from Structures_N import *

vecttest = np.zeros(n-md)



def OraclePG(qc,ind):
#    F = 0
#    G = np.zeros(n - md)
#    Bqc = np.dot(B,qc)
#    F = (1/3)*np.vdot(q0 + Bqc, (r*(q0 + Bqc)*np.abs(q0 + Bqc))) + np.vdot(pr, np.dot(Ar,q0 + Bqc))
#    G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
    if (ind == 2) :
        F = 0
        F = (1/3)*np.vdot(q0 + Bqc, (r*(q0 + Bqc)*np.abs(q0 + Bqc))) + np.vdot(pr, np.dot(Ar,q0 + Bqc))
        return(F)
    if (ind == 3):
        G = np.zeros(n - md)
        G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
        return (G)
    if (ind == 4):
        F = 0
        G = np.zeros(n - md)
        Bqc = np.dot(B,qc)
        F = (1/3)*np.vdot(q0 + Bqc, (r*(q0 + Bqc)*np.abs(q0 + Bqc))) + np.vdot(pr, np.dot(Ar,q0 + Bqc))
        G = np.dot(np.transpose(B), r*(q0 + Bqc)*np.abs(q0 + Bqc)) + np.dot(np.dot(np.transpose(B),np.transpose(Ar)),pr)
        return(F,G)
    
    
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    