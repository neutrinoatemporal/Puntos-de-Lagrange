import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from scipy.optimize import fsolve
from sympy import *
from sympy.plotting import plot
from sympy.abc import x

#Constantes
G= 6.673e-11 # Constante de gravitación universal (N m^2 kg^-2)
MS= 1.9891e30 # Masa solar (kg)
RS= 6.95508e8 # Radio solar (m)

## Primer sistema de coordenadas

def puntos_lagrange(m1, m2, a): # Función para hallar los primeros puntos de Lagrange y graficar las superfisies equipotenciales a las que pertenecen
    d=a 
    masa1= m1*MS  # Masa de la estrella 1
    masa2= m2*MS  # Masa de la estrella 2
    a= a*RS       # distancia entre estrellas
    r1= -a/(1+masa1/masa2)  # posición de la estrella 1
    r2= a/(1+masa2/masa1)   # posición de la estrella 2
    x = np.linspace(-2*a, 2*a, 10000)  
    we2= G*(masa1+masa2)/a**3  # velocidad angular al cuadrado

    ## PUNTOS DE LAGRANGE

    # Potencial gravitatorio
    V = np.where(x>=0,
                 -G*(masa1/( np.abs(np.sqrt(r1**2 + x**2 +2*np.abs(r1)*np.abs(x)))) + masa2/(np.abs(np.sqrt(r2**2 + x**2 -2*np.abs(r2)*np.abs(x)))))-(we2*x**2)/2,  # Para x < 0
                 -G*(masa1/(np.abs(np.sqrt(r1**2 + x**2 -2*np.abs(r1)*np.abs(x)))) + masa2/(np.abs(np.sqrt(r2**2 + x**2 +2*np.abs(r2)*np.abs(x)))))-(we2*x**2)/2 ) # Para x < 0


    x_a= x/ a # X en unidades de a
    V_G=V/(G*(masa1 +masa2)/a) # Potencial en unidades de G(m1+m2)/a

    plt.figure(figsize=(10, 6)) ## Gráfica Potencial Vs X
    plt.plot(x_a, V_G,color='black', label='Potencial gravitacional')
    plt.axvline(x=r1/a, color='r', linestyle='--', label='M1')
    plt.axvline(x=r2/a, color='b', linestyle='--', label='M2')
    plt.axhline(y=0, color='k', linestyle='-')
    plt.title(f'Potencial gravitacional (M1={m1} M, M2={m2} M, a={d} R)')
    plt.xlabel('x/a ')
    plt.ylabel('V/[G(M1+M2)/a]')
    plt.xlim(-2, 2)
    plt.ylim(-3, -1)
    x_ticks = np.arange(-2, 2.5, 0.5)
    y_ticks = np.arange(-3, -0.5, 0.5)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.legend()
    plt.grid(True)
    plt.show()

    dy_dx = np.gradient(V_G, x_a) ##Derivada del potencial respecto a X
    valid_indices = ~np.isnan(dy_dx) & ~np.isinf(dy_dx) # Identificar valores donde la derivada no está definida

    # Filtrar valores de X y de la derivada del potencial para mantener solo los valores definidos
    x_a_valido = x_a[valid_indices]
    dy_dx_valido = dy_dx[valid_indices]

    cambio_signo = np.where(np.diff(np.sign(dy_dx_valido)))[0] # Encontrar los puntos donde la derivada cambia de signo

    puntos_criticos = x_a_valido[cambio_signo + 1] # Los puntos críticos son los valores de x en los índices encontrados más 1

    d2y_dx2 = np.gradient(dy_dx_valido, x_a_valido) # Calcular la segunda derivada del potencial respecto a X

    lagrange = [] # Encontrar los máximos locales (donde la segunda derivada es negativa) Estos son los puntos de Lagrange
    for p in puntos_criticos:
        # Buscar el índice más cercano de x_a_valid a point
        idx = np.searchsorted(x_a_valido, p)
        # Asegurarse de que el índice está dentro de los límites
        if 0 < idx < len(d2y_dx2):
            # Verificar si la segunda derivada es negativa
            if d2y_dx2[idx] < 0:
                lagrange.append(p)


    ##CURVAS EQUIPOTENCIALES

    x = np.linspace(-2.5*a, 2.5*a, 500) 
    y = np.linspace(-2.5*a, 2.5*a, 500)  
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2) # radio polar
    theta = np.arctan2(Y, X) # angulo theta polar

    ##Potencial
    V=-G*(masa1/np.sqrt(r1**2+r**2+2*np.abs(r1)*np.abs(r)*np.cos(theta)) + masa2/np.sqrt(r2**2+r**2-2*np.abs(r2)*np.abs(r)*np.cos(theta)))- (we2*r**2)/2

    niveles_potencial = [] ## Vector para guardar los valores de potencial de los puntos de Lagrange
    print("Para los tres primeros Puntos de Lagrange r=(x,0), estan contenidos en las siguientes superficies equipotenciales:")
    for R in lagrange: ## Se evalua el potencial para cada punto de Lagrange
        print("Punto r= ", R)
        R=R*a
        if R<0:
            V_r=-G*(masa1/np.sqrt(r1**2+R**2-2*np.abs(r1)*np.abs(R)) + masa2/np.sqrt(r2**2+R**2+2*np.abs(r2)*np.abs(R)))- (we2*R**2)/2
        else:
            V_r=-G*(masa1/np.sqrt(r1**2+R**2+2*np.abs(r1)*np.abs(R)) + masa2/np.sqrt(r2**2+R**2-2*np.abs(r2)*np.abs(R)))- (we2*R**2)/2
        V_r_G = V_r / (G * (masa1 + masa2) / a) # Normalizar V_r
        print("V=", V_r_G)
        niveles_potencial.append(V_r_G) # Se guarda el potencial

    niveles_potencial= np.sort(niveles_potencial) # Se ordenan los valores de potencial


    V_G=V/(G*(masa1 +masa2)/a) # Potencial en unidades G*(masa1 +masa2)/a
    x_a=x/a # X en unidades de a
    y_a=y/a # Y en unidades de a
    X_a, Y_a = np.meshgrid(x_a, y_a)

    plt.figure(figsize=(8, 6)) ## Gráfica de los equipotenciales
    colores = ['purple', 'red', 'blue']
    lineas = []

    for i, nivel in enumerate(niveles_potencial):
        contorno = plt.contour(X_a, Y_a, V_G, levels=[nivel], colors=[colores[i % len(colores)]], linewidths=1, linestyles='solid')
        lineas.append(plt.Line2D([0], [0], color=colores[i], lw=2))

    plt.scatter([r1/a, r2/a], [0, 0], color='black', zorder=5)
    plt.annotate('M1', (r1/a, 0), textcoords="offset points", xytext=(10, 10), ha='center', color='black', fontsize=12)
    plt.annotate('M2', (r2/a, 0), textcoords="offset points", xytext=(10, 10), ha='center', color='black', fontsize=12)
    etiquetas = [f'$\Phi$ = {nivel:.2f}' for nivel in niveles_potencial]
    plt.legend(lineas, etiquetas, loc='upper right', fontsize=10, frameon=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x/a")
    plt.ylabel("y/a")
    plt.title('Superficies equipotenciales que contienen a los P.L')
    plt.grid(True)
    plt.show()


## Segundo sistema de coordenadas

## Hallar Punto L1

def newton_raphson(g, dg, x0, tol, max_iter=100): # Método de Newton-Raphson
    x = x0
    for i in range(max_iter):
        gx = g(x)
        dgx = dg(x)
        if abs(gx) < tol:  # Si el valor de g(x) es suficientemente pequeño, se ha encontrado la raíz
            return x  # Devuelve la raíz y el número de iteraciones
        if dgx == 0:  # Si la derivada es cero, no se puede aplicar el método
            raise ValueError("Derivada cero. No se puede continuar con Newton-Raphson.")
        x = x - gx / dgx  # Fórmula del método de Newton-Raphson
    raise ValueError("El método no convergió en el número máximo de iteraciones.")

def l1(q): #Punto de Lagrange L1

    def g(x): # Definir la función g(x)=dV(x)/dx y su derivada
        return -1/x**2 - q*(((x-1)/np.sqrt(x**2-2*x+1)**3)+1)+ (1+q)*x

    def dg(x):
        return 2/x**3+ 2*q/np.sqrt(x**2-2*x+1)**3 + (1+q)
   # Parámetros iniciales
    x0 = 0.5  # Suposición inicial
    tol = 1e-9 # Elegimos un diferencia menor a 10^-9 para detener la iteración

    raiz = newton_raphson(g, dg, x0, tol,100)
    return raiz

## Hallar radios polares de estrellas

def rp(q,p): # Radio Polar

    def f(r):
        return 1/r + q*(1/(np.sqrt(r**2+1)))- p

    def df(r):
        return -1/r**2- q*r/(r**2+1)**(3/2)

    def g(x):
        return -1/x**2 - q*(((x-1)/np.sqrt(x**2-2*x+1)**3)+1)+ (1+q)*x

    def dg(x):
        return 2/x**3+ 2*q/np.sqrt(x**2-2*x+1)**3 + (1+q)
    tol = 1e-9
    r0 = newton_raphson(g, dg, 0.5, tol,100)/3
    raiz = newton_raphson(f, df, r0, tol,100)

    return raiz

## Superficies equipotenciales

def equipotenciales(q):
    Z_vals = np.linspace(-0.6, 0.6, 200)
    X_vals = np.linspace(-0.9, 1.5, 200)
    Z, X = np.meshgrid(Z_vals, X_vals)

    r = np.sqrt(X**2 + Z**2)  # Distancia radial
    theta = np.pi/2- np.arctan2(Z, X)  # Ángulo con respecto al eje z
    lambda_prime = np.sin(theta)
    nu_prime=np.cos(theta)

    Pt = 1/r + q * ((1 / np.sqrt(r**2 - 2 * r * lambda_prime + 1))-r*lambda_prime) + (r**2)*(1+q)*(1-nu_prime**2)/2
    R = l1(q)
    Ptl1 = 1/R + q * ((1 / np.sqrt(R**2 - 2 * R + 1))-R) + ((R**2)*(1+q)/2)

    niveles_potencial = [Ptl1 - (Ptl1*q*0.1), Ptl1-(Ptl1*q*0.15), Ptl1, Ptl1 * 2, Ptl1 * 3]
    niveles_potencial= np.sort(niveles_potencial)

    plt.figure(figsize=(8, 6))
    colores = ['red', 'magenta', 'black', 'green', 'blue']
    lineas = []
    for i, nivel in enumerate(niveles_potencial):
        contorno = plt.contour(X, Z, Pt, levels=[nivel], colors=[colores[i % len(colores)]], linewidths=2)
        lineas.append(plt.Line2D([0], [0], color=colores[i], lw=2))

    plt.scatter([1, 0], [0, 0], color='black', zorder=5)
    plt.annotate('M1', (0, 0), textcoords="offset points", xytext=(10, 10), ha='center', color='black', fontsize=12)
    plt.annotate('M2', (1, 0), textcoords="offset points", xytext=(10, 10), ha='center', color='black', fontsize=12)
    etiquetas = [f'$\Omega$ = {nivel:.2f}' for nivel in niveles_potencial]
    plt.legend(lineas, etiquetas, loc='upper left', fontsize= 8, frameon=False)
    plt.xlabel("x/a")
    plt.ylabel("z/a")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Superficies Equipotenciales')
    plt.grid(True)
    plt.show()

    for p in niveles_potencial:
        print(f"Para q= {q} y potencial= {p}, Radio polar: {rp(q,p)}")