#!/usr/bin/python

import numpy as np

from numpy.linalg import norm
from time import process_time
from Wolfe_Skel import Wolfe

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Algorithme BFGS                                                   #
#                                                                           #
#############################################################################

from Visualg import Visualg
from Oracle import *


def BFGS(Oracle, x0):
    ##### Initialisation des variables

    iter_max = 10000
    # gradient_step_ini = 1.  # Problème primal.
    gradient_step_ini = 1000. # Problème dual.
    threshold = 0.000001

    error_count = 0 # Compteur de non-convergence de l'algorithme de Fletcher-Lemarechal.

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0

    ##### Boucle sur les iterations
    for k in range(iter_max):
        # Nouvelles valeurs du critere et du gradient
        critere, gradient = Oracle(x, 4)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        if k == 0:
            W = np.eye(len(gradient))
        else:
            delta_x = x - x_p
            delta_g = gradient - gradient_p
            delta_mat_1 = np.outer(delta_x, delta_g) / np.vdot(delta_g, delta_x)
            delta_mat_2 = np.outer(delta_x, delta_x) / np.vdot(delta_g, delta_x)
            I = np.eye(len(gradient)) # Matrice identité
            W = np.dot(np.dot(I - delta_mat_1, W_p), I - np.transpose(delta_mat_1)) + delta_mat_2
        direction =  np.dot(-W, gradient)

        # Pas de descente
        gradient_step, error_code = Wolfe(gradient_step_ini, x, direction, Oracle)

        if error_code != 1:
            error_count += 1

        # Mise a jour des variables
        x_p = x                          # Valeur précédente de la position
        gradient_p = gradient   # Valeur précédente du gradient
        direction_p = direction # Valeur précédente de la direction
        W_p = W
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