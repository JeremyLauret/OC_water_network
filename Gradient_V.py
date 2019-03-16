#!/usr/bin/python

import numpy as np

from numpy.linalg import norm
from time import process_time
from Wolfe_Skel import Wolfe

from Fletcher_step import fletcher_step

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode du gradient a pas fixe                                    #
#                                                                           #
#############################################################################

from Visualg import Visualg


def Gradient_V(Oracle, x0):
    ##### Initialisation des variables

    iter_max = 10000
    gradient_step_ini = 0.0005
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0

    ##### Boucle sur les iterations

    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient = Oracle(x, 4)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        D = -gradient

        # Pas de descente
        gradient_step, code = Wolfe(gradient_step_ini, x, D, Oracle)

        if code != 1:
            print("WARNING : the Fletcher-Lemarechal algorthm did not converge properly.")
            print("Gradient step : {}".format(gradient_step))

        # Mise a jour des variables
        x = x + (gradient_step * D)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

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
