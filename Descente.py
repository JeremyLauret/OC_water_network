#!/usr/bin/python

import numpy as np
from numpy.linalg import norm
from time import process_time
import matplotlib.pyplot as plt
from Wolfe_Skel import Wolfe

#############################################################################
#                                                                           #
#         RECHERCHE D'UNE DIRECTION DE DESCENTE                             #
#                                                                           #
#         Méthodes de Polak-Ribière, BFGS et Newton                         #
#                                                                           #
#############################################################################

from Visualg import Visualg
from Oracle import *


def Polak_Ribiere(Oracle, x0):
    ##### Initialisation des variables

    iter_max = 10000
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []
    gradient_list = []

    time_start = process_time()

    # Point courant
    x = x0

    #### Premiere iteration
    critere, gradient = Oracle(x, 4)
    gradient_norm = norm(gradient)

    # Direction de descente
    D = -gradient

    # delta_k = critere
    alpha = 1 ####################################### TO DO : Fletcher
    alpha = Wolfe(alpha, x, D, Oracle)
    x += alpha * D

    gradient_norm_list.append(gradient_norm)
    gradient_step_list.append(alpha)
    critere_list.append(critere)
    gradient_list.append(gradient)

    ##### Boucle sur les iterations

    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient = Oracle(x, 4)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break

        # Calcul de beta
        beta = np.dot(np.transpose(gradient), gradient - gradient_list[-1]) / gradient_norm_list[-1]**2

        # Direction de descente
        D = -gradient + beta * D

        # delta_k = critere
        alpha = 1  ####################################### TO DO : Fletcher
        alpha = Wolfe(alpha, x, D, Oracle)

        # Mise a jour des variables
        x += alpha * D

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(alpha)
        critere_list.append(critere)
        gradient_list.append(gradient)

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

    return critere_opt, gradient_opt, x_opt