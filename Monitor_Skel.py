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

# ---> Charger les fonctions associees a l'oracle du probleme,
#      aux algorithmes d'optimisation et de recherche lineaire
#
#      Exemple 1 - le gradient a pas fixe :
#
#                  from OraclePG import OraclePG
#                  from Gradient_F import Gradient_F
#
#      Exemple 2 - le gradient a pas variable :
#
#                  from OraclePG import OraclePG
#                  from Gradient_V import Gradient_V
#                  from Wolfe import Wolfe
#
# ---> A modifier...

from Oracle import OraclePG, OraclePH   # Oracles.
from Gradient_F import Gradient_F       # Gradient à pas fixe.
from Optim_Numpy import Optim_Numpy     # Fonction scipy.optimize.minimize.
from Newton_F import Newton_F           # Newton à pas fixe.
from Gradient_V import Gradient_V       # Gradient à pas variable.
from Polak_Ribiere import Polak_Ribiere # Gradient conjugué non linéaire.
from BFGS import BFGS                   # Quasi-Newton avec formule de BFGS.

##### Initialisation de l'algorithme

# ---> La dimension du vecteur dans l'espace primal est n-md
#      et la dimension du vecteur dans l'espace dual est md
#
#      Probleme primal :
#
#                        x0 = 0.1 * np.random.normal(size=n-md)
#
#      Probleme dual :
#
#                        x0 = 100 + np.random.normal(size=md)
#
# ---> A modifier...

x0 = 0.1 * np.random.normal(size = n - md) # Initialisation du problème primal.

##### Minimisation proprement dite

# ---> Executer la fonction d'optimisation choisie
#
#      Exemple 1 - le gradient a pas fixe :
#
#                  print()
#                  print("ALGORITHME DU GRADIENT A PAS FIXE")
#                  copt, gopt, xopt = Gradient_F(OraclePG, x0)
#
#      Exemple 2 - le gradient a pas variable :
#
#                  print()
#                  print("ALGORITHME DU GRADIENT A PAS VARIABLE")
#                  copt, gopt, xopt = Gradient_V(OraclePG, x0)
#
# ---> A modifier...

print()

# print("ALGORITHME DU GRADIENT A PAS FIXE")
# copt, gopt, xopt = Gradient_F(OraclePG, x0)

# print("ALGORITHME DE MINIMISATION DE SCIPY")
# copt, gopt, xopt = Optim_Numpy(OraclePG, x0)

# print("ALGORITHME DE NEWTON A PAS FIXE")
# copt, gopt, xopt = Newton_F(OraclePH, x0)

# print("ALGORITHME DE GRADIENT A PAS VARIABLE")
# copt, gopt, xopt = Gradient_V(OraclePH, x0)

# print("ALGORITHME DE POLAK RIBIERE")
# copt, gopt, xopt = Polak_Ribiere(OraclePH, x0)

print("ALGORITHME BFGS")
copt, gopt, xopt = BFGS(OraclePH, x0)

##### Verification des resultats

# ---> La fonction qui reconstitue les variables hydrauliques
#      du reseau a partir de la solution du probleme s'appelle
#      HydrauliqueP pour le probleme primal, et HydrauliqueD
#      pour le probleme dual
#
#      Probleme primal :
#
#                        qopt, zopt, fopt, popt = HydrauliqueP(xopt)
#
#
#                        qopt, zopt, fopt, popt = HydrauliqueDxopt)
#
# ---> A modifier...

qopt, zopt, fopt, popt = HydrauliqueP(xopt) # Problème primal.

Verification(qopt, zopt, fopt, popt)
