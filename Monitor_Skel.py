#!/usr/bin/python

import numpy as np

#############################################################################
#                                                                           #
#  MONITEUR D'ENCHAINEMENT POUR LE CALCUL DE L'EQUILIBRE D'UN RESEAU D'EAU  #
#                                                                           #
#############################################################################

##### Fonctions fournies dans le cadre du projet

# Donnees du probleme
from Probleme_R import *
from Structures_N import *

# Affichage des resultats
from Visualg import Visualg

# Verification des resultats
from HydrauliqueP import HydrauliqueP
from HydrauliqueD import HydrauliqueD
from Verification import Verification

##### Fonctions a ecrire dans le cadre du projet

from Oracle import OraclePG, OraclePH, OracleDG, OracleDH   # Oracles.
from Gradient_F import Gradient_F                           # Gradient à pas fixe.
from Optim_Numpy import Optim_Numpy                         # Fonction scipy.optimize.minimize.
from Newton_F import Newton_F                               # Newton à pas fixe.
from Gradient_V import Gradient_V                           # Gradient à pas variable.
from Polak_Ribiere import Polak_Ribiere                     # Gradient conjugué non linéaire.
from BFGS import BFGS                                       # Quasi-Newton avec formule de BFGS.
from Newton_V import Newton_V                               # Newton à pas variable.

##### Initialisation de l'algorithme

# x0 = 0.1 * np.random.normal(size = n - md) # Initialisation du problème primal.
x0 = 100 + np.random.normal(size = md) # Initialisation du problème dual.

##### Minimisation proprement dite

print()

## ---- Problème Primal ----
# print("ALGORITHME DU GRADIENT A PAS FIXE (PROBLEME PRIMAL)")
# copt, gopt, xopt = Gradient_F(OraclePG, x0)

# print("ALGORITHME DE MINIMISATION DE SCIPY (PROBLEME PRIMAL)")
# copt, gopt, xopt = Optim_Numpy(OraclePG, x0)

# print("ALGORITHME DE NEWTON A PAS FIXE (PROBLEME PRIMAL)")
# copt, gopt, xopt = Newton_F(OraclePH, x0)

# print("ALGORITHME DE GRADIENT A PAS VARIABLE (PROBLEME PRIMAL)")
# copt, gopt, xopt = Gradient_V(OraclePH, x0)

# print("ALGORITHME DE POLAK RIBIERE (PROBLEME PRIMAL)")
# copt, gopt, xopt = Polak_Ribiere(OraclePH, x0)

# print("ALGORITHME BFGS (PROBLEME PRIMAL)")
# copt, gopt, xopt = BFGS(OraclePH, x0)

# print("ALGORITHME DE NEWTON A PAS VARIABLE (PROBLEME PRIMAL)")
# copt, gopt, xopt = Newton_V(OraclePH, x0)


## ---- Problème Dual ----
# print("ALGORITHME DU GRADIENT A PAS FIXE (PROBLEME DUAL)")
# copt, gopt, xopt = Gradient_F(OracleDG, x0)

# print("ALGORITHME DE MINIMISATION DE SCIPY (PROBLEME DUAL)")
# copt, gopt, xopt = Optim_Numpy(OracleDG, x0)

# print("ALGORITHME DE NEWTON A PAS FIXE (PROBLEME DUAL)")
# copt, gopt, xopt = Newton_F(OracleDH, x0)

# print("ALGORITHME DE GRADIENT A PAS VARIABLE (PROBLEME DUAL)")
# copt, gopt, xopt = Gradient_V(OracleDH, x0)

# print("ALGORITHME DE POLAK RIBIERE (PROBLEME DUAL)")
# copt, gopt, xopt = Polak_Ribiere(OracleDH, x0)

# print("ALGORITHME BFGS (PROBLEME DUAL)")
# copt, gopt, xopt = BFGS(OracleDH, x0)

print("ALGORITHME DE NEWTON A PAS VARIABLE (PROBLEME DUAL)")
copt, gopt, xopt = Newton_V(OracleDH, x0)

##### Verification des resultats

# qopt, zopt, fopt, popt = HydrauliqueP(xopt) # Problème primal.
qopt, zopt, fopt, popt = HydrauliqueD(xopt)  # Problème dual.

Verification(qopt, zopt, fopt, popt)
