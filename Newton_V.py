#!/usr/bin/python

import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.linalg import inv
from time import process_time
from Wolfe_Skel import Wolfe

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode de Newton a pas variable                                  #
#                                                                           #
#############################################################################

from Visualg import Visualg
from Oracle import *


def Newton_V(Oracle, x0):
    ##### Initialisation des variables

    iter_max = 100
    # gradient_step_ini = 1. # Problème primal.
    gradient_step_ini = 1000  # Problème dual.
    threshold = 0.000001

    error_count = 0  # Compteur de non-convergence de l'algorithme de Fletcher-Lemarechal.

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0

    ##### Boucle sur les iterations
    for k in range(iter_max):
        # Valeur du critere et du gradient
        critere, gradient, hessien = Oracle(x, 7)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        direction = - dot(inv(hessien), gradient)

        # Pas de descente
        gradient_step, error_code = Wolfe(gradient_step_ini, x, direction, Oracle)

        if error_code != 1:
            error_count += 1

        # Mise a jour des variables
        x = x + (gradient_step * direction)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

    if error_count > 0:
        print()
        print("Non-convergence de l'algorithme de Fletcher-Lemarechal : {}".format(error_count))

    ##### Resultats de l'optimisation
    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt,
