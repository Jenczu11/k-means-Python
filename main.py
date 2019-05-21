import argparse
import textwrap

from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd


def saveToFile(filename, tryb, how, iterations, error_iteration_list):
    file = open(filename, tryb)
    file.write("\n---pow:-" + repr(how) + "--------")
    file.write("\nilosc epok " + repr(iterations))
    file.write("\nbledy epokami\n")
    file.write(str(error_iteration_list))
    file.write("\nOstatni blad kwadratowy " + repr(error_iteration_list[len(error_iteration_list) - 1]))
    file.write("\n-----KONIEC----------")
    file.close()


def initParser():
    # Obsluga przez parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
		 GRUPOWANIE
Uzywany alogrytm kNN
Liczba obserwacji: 1000

Atrybuty (kolumny):
1. współrzędna x
2. współrzędna y

			 '''))
    # parser.add_argument('n', help='Ilosc atrybutow', type=int)
    # parser.add_argument('ATTRIBUTES', type=int, metavar='N', nargs='+',
    #                     help="Podaj po spacji ktore atrybuty chcesz analizowac")
    parser.add_argument('clusterNumber', type=int, help='Podaj ile chcesz clustrow')
    parser.add_argument('howManyTimes', type=int, help='Ile razy ma sie wykonac')
    # parser.add_argument('-v', '--verbose', help="Tryb debug", action='count', default=0)
    parser.add_argument('-w', '--wykres', help="Czy chcesz wykresy", action='store_true')
    # parser.add_argument('-aN', '--allNeighbors', help="Wszyscy sasiedzi [3,5,itd]", action='store_true')

    return parser.parse_args()


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def main():
    args = initParser()
    verbose = False
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    data = pd.read_csv("data.csv", names=["X", "Y"])
    if verbose: print(data.shape)

    # print(data)
    for how in range(args.howManyTimes):
        f1 = data['X'].values
        if verbose: print("f1")
        f2 = data['Y'].values
        if verbose: print("f2")
        X = np.array(list(zip(f1, f2)))
        if verbose: print(X)
        if verbose: print("stworzylemX")
        # plt.scatter(f1, f2, c='black', s=7)
        # plt.show()
        if verbose: print("Pierwszy wykres")
        lista = []
        # WARNING
        # Number of clusters
        print(args.clusterNumber)
        k = args.clusterNumber
        # k = 25
        # X coordinates of random centroids
        C_x = np.random.randint(0, np.max(X), size=k)
        if verbose: print("C_x")
        # Y coordinates of random centroids
        C_y = np.random.randint(0, np.max(X), size=k)
        if verbose: print("C_y")
        C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
        if verbose: print("Stworzylem C")
        if verbose: print(C)
        # plt.scatter(f1, f2, c='#050505', s=7)
        # plt.scatter(C_x, C_y, marker='*', s=200, c='g')
        # plt.show()
        # print("Drugi wykres")
        # To store the value of centroids when it updates
        C_old = np.zeros(C.shape)
        # Cluster Lables(0, 1, 2)
        clusters = np.zeros(len(X))
        # Error func. - Distance between new centroids and old centroids
        error = dist(C, C_old, None)
        # Loop will run till the error becomes zero
        iterations = 0;
        error_iteration_list = []
        while error != 0:
            iterations = iterations + 1
            error_1 = 0
            # Assigning each value to its closest cluster
            for i in range(len(X)):
                distances = dist(X[i], C)
                cluster = np.argmin(distances)
                clusters[i] = cluster
                temp_1 = pow(X[i][0] - C[cluster][0], 2)
                temp_2 = pow(X[i][1] - C[cluster][1], 2)
                error_1 += temp_1 + temp_2
            if verbose: print("dziele")
            error_1 = error_1 / 1000
            if verbose: print("dodaje")
            error_iteration_list.append(error_1)
            # tutaj zrobic wg wzoru i na koncu podzielic przez 1000 powinien wyjsc blad
            # Storing the old centroid values
            C_old = deepcopy(C)  # deepcopy -> to kopia nie wskaznik
            # print(clusters)
            # zrobic 3 listy punkt i centroid [xi,yi, cx,cy] odpowiednio przyporzadkowane
            # Finding the new centroids by taking the average value
            for i in range(k):
                # i leci od 0,1,2 a points to X[j] gdzie j=0..999
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                # jezeli cluster nie dostal punktow to dostaje random z danych
                if len(points) == 0:
                    if verbose: print("EMPTY")
                    #points=[0]
                    points = X[np.random.randint(0, np.max(X), size=k)]
                C[i] = np.mean(points, axis=0)

            # odleglosc miedzy starym a nowym centroidem jezeli 0 to skoncz program
            error = dist(C, C_old, None)
        # print(error)

        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'bisque', 'forestgreen', 'cornflowerblue', 'brown', 'orange', 'lime',
                  'royalblue'
            , 'wheat', 'navy']
        fig, ax = plt.subplots()
        from collections import OrderedDict

        cmaps = OrderedDict()  # <- new colors
        cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper', 'Purples', 'Blues',
            'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
            'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
            'BuGn', 'YlGn']
        if args.wykres:
            for i in range(k):
                points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
                # ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i]) <- old colors
                if len(points) == 0:  # <- cheat bo znowu sie nie chcialo wykonac z tym dziala xdd
                    # jezeli zbior pusty to dalej
                    if verbose: print("empty centroid")
                    continue
                # take all points scatter(x,y, s=? [scalar/size of points], cmap/c=[color])
                # czy zmieniamy kolory
                ax.scatter(points[:, 0], points[:, 1], s=25, cmap=cmaps)
            # take final centroids and place them on plot with a * or d - for diamond sign and black color
            # diffrent markers at https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
            ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#000000')
            plt.show()
        print("\n---pow:-" + repr(how) + "--------")
        #print("ilosc epok " + repr(iterations))
        #print("bledy epokami")
        #print(error_iteration_list)
        print("Ostatni blad kwadratowy " + repr(error_iteration_list[len(error_iteration_list) - 1]))
        #rint("-----KONIEC----------")
        saveToFile("output.txt", "a", how, iterations, error_iteration_list)


# print(len(bledy_kwa_epokami))
# print("ilosc prob: "+repr(ilosc_prob))
main()
