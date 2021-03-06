#!/usr/bin/python

import numpy as np

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
    critere, gradient = Oracle(x, 4)

    # Produit scalaire du gradient initial avec la direction de descente
    gradient_scal_D = np.vdot(gradient, D)
    
    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x
    
    # Boucle de calcul du pas
    while ok == 0:
        xp = xn              # Point précédent.
        xn = x + alpha_n * D # Point actuel.
        critere_n, gradient_n = Oracle(xn, 4) # Critère et gradient actuels.

        # Calcul des conditions de Wolfe
        C1 = critere + omega_1 * alpha_n * gradient_scal_D - critere_n
        C2 = np.vdot(gradient_n, D) - omega_2 * gradient_scal_D

        # Test des conditions de Wolfe
        if C1 < 0: # Première condition de Wolfe non vérifiée.
            alpha_max = alpha_n
            alpha_n = 0.5 * (alpha_min + alpha_max)
        elif C2 < 0: # Seconde condition de Wolfe non vérifiée.
            alpha_min = alpha_n
            if alpha_max == np.inf:
                alpha_n = 2 * alpha_min
            else:
                alpha_n = 0.5 * (alpha_min + alpha_max)
        else:
            ok = 1
            break

        # Test d'indistinguabilite
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok
