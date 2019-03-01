#!/usr/bin/python

import numpy as np
#from numpy import dot
from Oracle import *
from Probleme_R import *
from Structures_N import *
vecttest = np.zeros(n-md)
vecttest[3] = 3
gradtest = [1,0,0,1,-6,1.6,0.34,1,1]

########################################################################
#                                                                      #
#          RECHERCHE LINEAIRE SUIVANT LES CONDITIONS DE WOLFE          #
#                                                                      #
#          Algorithme de Fletcher-Lemarechal                           #
#                                                                      #
########################################################################

#  Arguments en entree
#
#    alpha  : valeur initiale du pas
#    x      : valeur initiale des variables
#    D      : direction de descente
#    Oracle : nom de la fonction Oracle
#
#  Arguments en sortie
#
#    alphan : valeur du pas apres recherche lineaire
#    ok     : indicateur de reussite de la recherche 
#             = 1 : conditions de Wolfe verifiees
#             = 2 : indistinguabilite des iteres

def Wolfe(alpha, x, D, Oracle):
    
    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    ##### Algorithme de Fletcher-Lemarechal
    
    # Appel de l'oracle au point initial
    argout = Oracle(x,4)
    critere = argout[0]
    gradient = argout[1]
    
    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x
    
    # Boucle de calcul du pas
    while ok == 0:
        
        # xn represente le point pour la valeur courante du pas,
        # xp represente le point pour la valeur precedente du pas.
        xp = xn
        xn = x + np.asarray(alpha_n)*D  #np.asarray ajouté
        
        # Calcul des conditions de Wolfe
        argout_n = Oracle(xn,4)
        critere_n = argout_n[0]
        gradient_n = argout_n[1]
        argout_p = Oracle(xp,4)
        critere_p = argout_p[0]
        gradient_p = argout_p[1]
    
        C1 = (critere_n - critere_p) - (omega_1*alpha_n*np.vdot(gradient_p,D))
        C2 = (np.vdot(gradient_n,D)) - (omega_2*np.vdot(gradient_p,D))
        print(C1,C2,alpha_n)
        # Test des conditions de Wolfe
        if (C1 <= 0):
            if (C2 >= 0):
                ok = 1
            else :
                alpha_min = alpha_n
                if (alpha_max == np.inf):
                    alpha_n = 2*alpha_n
                else :
                    alpha_n = 0.5*(alpha_min + alpha_max)
        else :
            alpha_max = alpha_n
            alpha_n = 0.5*(alpha_min + alpha_max)
            
        # - si les deux conditions de Wolfe sont verifiees,
        #   faire ok = 1 : on sort alors de la boucle while
        # - sinon, modifier la valeur de alphan : on reboucle.
        #
        # ---> A completer...
        # ---> A completer...
        
        # Test d'indistinguabilite
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok
