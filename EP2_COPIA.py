import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random

'''Exercício Programa 2 - MAP3121 (2020)

    Autoria:
    Mariana Nakamoto - 10769793
    Thales Arantes Kerche Nunes - 10769372'''


def sol_aprox_u(delta_x, delta_t, N, M, u0, g1, g2, f):   # Função do EP1
    
    ti, xi = vetores_ti_xi(delta_t, delta_x, M, N)
    
    u = cria_u(M, N)
    
    u = implementa_u0(u, delta_x, u0)
    
    u = implementa_g1_g2(u, delta_t, g1, g2)

    for k in range (M):
        for i in range (1, N):
            u[k+1][i] = u[k][i] + delta_t*((u[k][i-1] - 2*u[k][i] + u[k][i+1])/(delta_x)**2 + f(ti[k], xi[i]))

    return u


def sol_exata_u(delta_x, delta_t, N, M, u0, g1, g2):   # Função do EP1

    ti, xi = vetores_ti_xi(delta_t, delta_x, M, N)
    
    u = cria_u(M, N)

    u = implementa_u0(u, delta_x, u0)
    
    u = implementa_g1_g2(u, delta_t, g1, g2)

    for k in range (M+1):
        for i in range (N+1):  # calcula os valores da formula exata, e os coloca na matriz U exata
            if u0 == u0_a:
                u[k][i] = (1 + math.sin(10*ti[k]))*(xi[i]**2)*(1 - xi[i])**2
            else:
                u[k][i] = math.exp(ti[k] - xi[i]) * math.cos(5*ti[k]*xi[i])
    return u


def err_u(u_exato, u_aprox, N):   # Função do EP1
    
    err_T = 0

    for i in range (N):
        err_ki = math.fabs(u_exato[-1][i] - u_aprox[-1][i])
        if err_ki > err_T:
            err_T = err_ki    
        
    return err_T


def euler(delta_x, N, u0, g1, g2, f):   # Função do EP1
    
    ti, xi = vetores_ti_xi(delta_x, delta_x, N, N)
    
    u = cria_u(N, N)

    u = implementa_u0(u, delta_x, u0)
        
    u = implementa_g1_g2(u, delta_x, g1, g2)
    
    for k in range (N):
        
        A = cria_A(1+2*N, (-1)*N)

        b = cria_b_euler(u, delta_x, f, N, g1, g2, ti, xi, k)
    
        ux = resolve_x_vetor(A, b)
        
        u = completa_u_euler(u, delta_x, f, N, ti, xi, k, ux)
       
    return u


def crank(delta_x, N, u0, g1, g2, f):   # Calcula a matriz U, para que possamos obter uk
                                        # e resolver o problema inverso
    ti, xi = vetores_ti_xi(delta_x, delta_x, N, N)
    
    u = cria_u(N, N)
    
    u = implementa_u0(u, delta_x, u0)
        
    u = implementa_g1_g2(u, delta_x, g1, g2)
    
    for k in range (N):
        
        A = cria_A(1+N, (-1)*N/2)
    
        b = cria_b_crank(u, delta_x, f, N, g1, g2, ti, xi, k)
    
        ux = resolve_x_vetor(A, b)
        
        u = completa_u_crank(u, delta_x, f, N, ti, xi, k, ux)
       
    return u


def vetores_ti_xi(delta_t, delta_x, M, N):   # Função do EP1
    
    ti = []
    xi = []
    for a in range (M+1):
        tk = a * delta_t
        ti.append(tk)
    for c in range (N+1):
        xk = c * delta_x
        xi.append(xk)
    
    return ti, xi


def cria_u(M, N):   # Função do EP1
    
    u = [0] *(M+1)
    for b in range (len(u)):
        u[b] = [0]* (N+1)
    
    return u


def implementa_u0(u, delta_x, u0):   # Função do EP1
    
    for a in range (1, len(u[0])):
        xh = a * delta_x
        u[0][a] = u0(xh)
        
    return u


def implementa_g1_g2(u, delta_t, g1, g2):   # Função do EP1
    
    for y in range(len(u)):
        th = y * delta_t
        u[y][0] = g1(th)
        u[y][N] = g2(th)
    
    return u


def cria_A(diag, sub_diag):   # Função do EP1
    
    A = [[diag]*(N)]
    A.append([sub_diag]*(N-1))

    return A


def cria_b_euler(u, delta_t, f, lamb, g1, g2, ti, xi, k):   # Função do EP1
    
    b = []
    bi = u[k][1]+delta_t*(f(ti[k+1], xi[1])) + lamb*(g1(ti[k+1]))
    b.append(bi)
    for i in range (2, N-1):
        bn = u[k][i]+delta_t*(f(ti[k+1], xi[i]))
        b.append(bn)
    bu = u[k][N-1]+delta_t*(f(ti[k+1], xi[N-1])) + lamb*(g2(ti[k+1]))
    b.append(bu)
    
    return b


def cria_b_crank(u, delta_t, f, lamb, g1, g2, ti, xi, k):   # Função do EP1 (necessária para calcular Crank-Nicolson no EP2)
    
    b = []
    bi = u[k][1]+delta_t/2*(f(ti[k+1], xi[1]) + f(ti[k], xi[1])) + lamb/2*(g1(ti[k+1])) \
         + lamb/2*(u[k][0]-2*u[k][1]+u[k][2])
    b.append(bi)
    for i in range (2, N-1):
        bn = u[k][i]+delta_t/2*(f(ti[k+1], xi[i]) + f(ti[k], xi[i])) + lamb/2*(u[k][i-1]-2*u[k][i]+u[k][i+1])
        b.append(bn)
    bu = u[k][N-1]+delta_t/2*(f(ti[k+1], xi[N-1]) + f(ti[k], xi[N-1])) + lamb/2*(g2(ti[k+1])) \
         + lamb/2*(u[k][N-2]-2*u[k][N-1]+u[k][N])
    b.append(bu)
    
    return b


def completa_u_euler(u, delta_t, f, lamb, ti, xi, k, ux):   # Função do EP1
    
    for r in range (1, N):
        if r == 1:
            u[k+1][r] = u[k][r] + lamb*(u[k+1][0]-2*ux[r-1]+ux[r]) + delta_t*(f(ti[k+1], xi[r]))
        elif r == N-1:
            u[k+1][r] = u[k][r] + lamb*(ux[r-2]-2*ux[r-1]+u[k+1][N]) + delta_t*(f(ti[k+1], xi[r]))
        else:
            u[k+1][r] = u[k][r] + lamb*(ux[r-2]-2*ux[r-1]+ux[r]) + delta_t*(f(ti[k+1], xi[r]))
    
    return u
       

def completa_u_crank(u, delta_t, f, lamb, ti, xi, k, ux):   # Função do EP1 (necessária para calcular Crank-Nicolson no EP2)
    
    for r in range (1, N):
        if r == 1:
            u[k+1][r] = u[k][r] + lamb/2*((u[k+1][0]-2*ux[r-1]+ux[r])+(u[k][r-1]-2*u[k][r]+u[k][r+1])) \
                        + delta_t/2*(f(ti[k], xi[r]) + f(ti[k+1], xi[r]))
        elif r == N-1:
            u[k+1][r] = u[k][r] + lamb/2*((ux[r-2]-2*ux[r-1]+u[k+1][N])+(u[k][r-1]-2*u[k][r]+u[k][r+1])) \
                        + delta_t/2*(f(ti[k], xi[r]) + f(ti[k+1], xi[r]))
        else:
            u[k+1][r] = u[k][r] + lamb/2*((ux[r-2]-2*ux[r-1]+ux[r])+(u[k][r-1]-2*u[k][r]+u[k][r+1])) \
                        + delta_t/2*(f(ti[k], xi[r]) + f(ti[k+1], xi[r]))
                        
    return u


def decompoe_A_vetor(A):   # Função do EP1

    D = [0] * len(A[0])
    L = [0] * len(A[1])   
    D[0] = A[0][0]

    for i in range (len(A[1])):
        L[i] = A[1][i]/D[i]
        D[i+1] = A[0][i+1] - (L[i]**2) * D[i]
    
    return D, L


def resolve_x_vetor(A, b):   # Função do EP1

    D, L = decompoe_A_vetor(A)
    
    z = [0] * len(b)
    c = [0] * len(b)
    x = [0] * len(b)
    
    z[0] = b[0]
    for i in range (1, len(z)):
        z[i] = b[i] - L[i-1] * z[i-1]

    for i in range (len(c)):
        c[i] = z[i]/D[i]

    x[-1] = c[-1]
    for i in range (len(x)-2, -1, -1):
        x[i] = c[i] - L[i]*x[i+1]
    
    return x


def escolhe_p(escolha):   # Como pedido no item 'a' essa função escolhe quais os valores de p
                          # dependendo do teste que está sendo realizado
    if escolha == 'a': p_list, uT_c = [0.35], 0                     # Simples, coloca a lista de p com valores do enunciado
    elif escolha == 'b': p_list, uT_c = [0.15, 0.3, 0.7, 0.8], 0    # (uT_c só é utilizado para os testes 'c' e 'd')
    else:
        with open ('teste.txt', 'r') as file:                       # Para testes 'c' e 'd', o arquivo 'teste.txt' precisa
            text = file.readlines()                                 # ser lido e transformado em uma matriz com floats

        text = [num.strip(' \n') for num in text]                   # retira os espaços em branco e quebra de linhas
        p_list = text.pop(0).split()                                # pega a primeira linha, valores de p, do arquivo
        p_list = [float(p) for p in p_list]                         # transforma esses valores em floats
        text = text[::int(2048/N)]                                  # pega a quantidade de pontos dependendo de N
        text.pop(0)                                                 # tira primeiro zero
        text.pop()                                                  # tira último zero
        uT_c = [float(value[:-5])/100 if 'E' in value else float(value) for value in text]   # vetor utilizado em 'c' e 'd'
        del text             # deleta o arquivo que não será mais usado                      # transformando em floats
                             # libera parte da RAM e melhora o desempenho
    return p_list, uT_c


def prod_int(u, v):                                                 # Calcula o produto interno para dois vetores dados
    
    total = 0
    for i in range (len(u)):
        z = u[i] * v[i]                                             # Faz a multiplicação dos valores de cada vetor
        total += z
    
    return total


def monta_sistema_min_qua(dict_p_u, uT):         # Monta o sistema pedido, com os produtos internos de cada vetor do teste
    
    matriz_final = [[0] * len(dict_p_u) for n in range (len(dict_p_u))]   # Faz matriz de zeros para ser alterada
    solucao = [0] * len(dict_p_u)                                         # Cria o vetor solução
    for i, u1 in enumerate(dict_p_u.values()):                            # Itera todos os vetores de todos os 'p'
        for j, u2 in enumerate(dict_p_u.values()):                        # Itera novamente para obter a matriz completa
            matriz_final[i][j] = prod_int(u1, u2)
        solucao[i] = prod_int(uT, u1)

    return matriz_final, solucao

def decompoe_A_matriz(A):                                               # Faz a decomposição de uma matriz simétrica
    
    D = [[0] * len(A[0]) for n in range (len(A))]                       # Cria matriz D de zeros
    L = [[0] * len(A[0]) for n in range (len(A))]                       # Cria matriz L de zeros
    for n in range (len(A)):                                            # Faz valores da diagonal de L igual 1
        L[n][n] = 1

    for j in range (len(A)):                                            # Decomposição da matriz A em LDLt
        for i in range (j):
            h = A[j][i]
            L[j][i] = h/D[i][i]
            for k in range (i+1, j+1):
                A[j][k] = A[j][k] - h*L[k][i]
        D[j][j] = A[j][j]
    
    return D, L


def resolve_x_matriz(A, b):                                             # Resolve sistema normal com A simétrica
    
    D, L = decompoe_A_matriz(A)                                         # Faz a decomposição LDLt
    
    z = [0] * len(A)
    c = [0] * len(A)
    for j in range(len(A)):                                             # Resolve primeiro sistema
        z[j] = b[j]
        for i in range (j):
            z[j] = z[j] - L[j][i]*z[i]
        c[j] = z[j]/D[j][j]

    x = [0] * len(A)
    for j in range (len(A)-1, -1, -1):                                  # Resolve segundo sistema
        x[j] = c[j]
        for i in range (j+1, len(A)):
            x[j] = x[j] - L[i][j]*x[i]
    
    return x


def erro_quad(delta_x, dict_p_u, uT, ak):                               # Calcula o erro quadrático de um teste
    
    somatorio = 0
    for i in range(len(uT)):
        fonte = 0
        for k, ui in enumerate(dict_p_u):                               # Resolve a fórmula dada no enunciado
            fonte += ak[k]*dict_p_u[ui][i]
        somatorio += (uT[i] - fonte)**2
    
    return math.sqrt(delta_x*somatorio)


def monta_plot(u_kTxi_all, uT):                                         # Funçao para calcular a estimativa com ak
                                                                        # e retornar o valor de uT aproximado
    plot_aprox = []
    for i in range(len(uT)):
        ux = 0
        for k, ui in enumerate(u_kTxi_all):
            ux += ak[k]*u_kTxi_all[ui][i]
        plot_aprox.append(ux)
    return plot_aprox


def implementa_ep2(escolha, delta_x, N, u0_c, g1_c, g2_c, f_p):         # Função que chama as outras, executa o EP2

    p_list, uT_c = escolhe_p(escolha)                                   # Primeiramente escolhe os valores de p
    u_kTxi_all = {}                                                     # de acordo com a escolha do teste
    global p                                                            # para que o 'p' vire variável global
    for p in p_list:                                                    # Calcula Crank para todos os 'p' da lista
            u_kTxi = crank(delta_x, N, u0_c, g1_c, g2_c, f_p)[-1]
            u_kTxi.pop(0)
            u_kTxi.pop()
            u_kTxi_all[p] = u_kTxi
    if escolha == 'a': uT = 7*np.array(u_kTxi_all[p_list[0]])           # Função dada, igual na letra 'b'
    elif escolha == 'b': uT = 2.3*np.array(u_kTxi_all[p_list[0]]) + 3.7*np.array(u_kTxi_all[p_list[1]]) \
                        + 0.3*np.array(u_kTxi_all[p_list[2]]) + 4.2*np.array(u_kTxi_all[p_list[3]])
    elif escolha == 'c': uT = uT_c
    else: uT = [elem*(1+((random.random() - 0.5)*2)*0.01) for elem in uT_c]    # Adiciona o fator alatório do teste 'd'
    sist, sol = monta_sistema_min_qua(u_kTxi_all, uT)                   # Monta o sistema dos minimos quadrados
    ak = resolve_x_matriz(sist, sol)                                    # Resolve o sistema e retorna lista de intensidades
    E2 = erro_quad(delta_x, u_kTxi_all, uT, ak)                         # Com a lista de intensidades, calcula o erro (E2)
    
    return ak, E2, u_kTxi_all, uT


def plot_graf2d(x, y, title, xlabel='N', ylabel='Erro Máximo'):         # Função do EP1 (usada para plotar resultados no EP2)
    
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_graf3d(u, T, N, M, lamb):                                      # Função do EP1
    
    x = np.linspace(0, 1, (N+1))
    t = np.linspace(0, T, (M+1))
    
    graf = plt.figure()
    ax = Axes3D(graf)
    x, t = np.meshgrid(x, t)
    u = np.array(u)
    ax.plot_surface(x, t, u, cmap=cm.get_cmap('Spectral'), linewidth=0)
    ax.set_xlabel("Posição")
    ax.set_ylabel("Tempo")
    ax.set_zlabel("Temperatura")
    plt.title(f"N = {N} e Lambda = {lamb}")
    plt.show()


def print_line(lenght=50):                                              # Função para a estética do cabeçalho, não importante
    print('=' *lenght)

def print_title(title, lenght=50):                                      # Função para a estética do cabeçalho, não importante
    size = len(title)
    final = (' ' *((lenght-size)//2)) + title + (' ' *((lenght-size)//2))
    print(final)

def header (title):                                                      # Função para a estética do cabeçalho, não importante
    print_line()
    print_title(title)
    print_line()

def menu (option_list, start=1):                                        # Função para a estética do cabeçalho, não importante
    for a in range (len(option_list)):
        print(f'{a+start} -- {option_list[a]}')



if __name__ == "__main__":                                              # Interface do programa

    print("Exercício Programa 2 - MAP3121")
    print("Autoria:    Mariana Nakamoto - 10769793")
    print("            Thales Arantes Kerche Nunes - 10769372\n")
    header("TESTES PARA A PARTE 1")
    menu(["Plotar U(t,x) e Erro x N para algum N máximo (a)",
          "Plotar U(t,x) e Erro x N para algum N máximo (b)",
          "Plotar U(t,x) para um N (c)\n"])
    header("TESTES PARA A PARTE 2")
    menu(["Plotar U(t,x) e Erro x N, por Euler Implícito, para algum N máximo (1a)",
          "Plotar U(t,x) e Erro x N, por Euler Implícito, para algum N máximo (1b)",
          "Plotar U(t,x), por Euler Implícito, para um N (1c)",
          "Plotar U(t,x) e Erro x N, por Crank-Nicolson, para algum N máximo (1a)",
          "Plotar U(t,x) e Erro x N, por Crank-Nicolson, para algum N máximo (1b)",
          "Plotar U(t,x), por Crank-Nicolson, para um N (1c)\n"], start=4)
    header("TESTE PARA O EP2 --- 10")
    menu(["Calcular as intensidades ak das fontes, e o erro quadrático E2\n"], start=10)
    
    while True:                             # programa insiste que o usuário digite um número válido, sem parar de rodar o programa
        try:
            teste = int(input("Qual teste deseja realizar? "))
            N_max = int(input("Digite o N a ser utilizado no teste: "))
        except:
            print("\033[31mERRO! Digite um número inteiro válido\033[m")
        else:
            break

    
    T = 1
    u0_a = lambda x: x**2*(1 - x)**2                                    # Todas funções do EP1, menos a última (f_p)
    g1_a = lambda t: 0
    g2_a = lambda t: 0
    f_a = lambda t, x: 10*math.cos(10*t)*(x**2)*(1 - x)**2 - (1 + math.sin(10*t)) * (12*(x**2) - 12*x + 2)
    u0_b = lambda x: math.exp((-1)*x)
    g1_b = lambda t: math.exp(t)
    g2_b = lambda t: math.exp(t-1)*math.cos(5*t)
    f_b = lambda t, x: 5*math.exp(t-x)*(5*(t**2)*math.cos(5*t*x) - (2*t+x)*math.sin(5*t*x))
    u0_c = lambda x: 0
    g1_c = lambda t: 0
    g2_c = lambda t: 0
    f_c = lambda t, x: 10000*(1 - 2*(t**2)) * (1/delta_x) \
                       if x >= (0.25 - delta_x/2) and x <= (0.25 + delta_x/2) else 0
    f_p = lambda t, x: 10*(1 + math.cos(5*t)) * (1/delta_x) \
                       if x >= (p - delta_x/2) and x <= (p + delta_x/2) else 0
    
    inf = {'a': [u0_a, g1_a, g2_a, f_a], 'b': [u0_b, g1_b, g2_b, f_b], 'c': [u0_c, g1_c, g2_c, f_c]}  # Dados para cada letra (EP1)
    
    if teste <= 3:                                                      # Testes do EP1
        lamb = float(input("Digite o Lambda a ser utilizado no teste: "))
        
        if not teste == 3:
            N = 10
            Ni = []
            err_max = []
            while (N <= N_max):
                delta_x = 1/N
                delta_t = (delta_x)**2 * lamb
                M = round(T/delta_t)
                if teste == 1: caso = 'a'
                else: caso = 'b'
                u_aprox = sol_aprox_u(delta_x, delta_t, N, M, *inf[caso])
                u_exato = sol_exata_u(delta_x, delta_t, N, M, *inf[caso][:-1])
                if N == N_max:
                    plot_graf3d(u_aprox, T, N, M, lamb)
                err_N = err_u(u_exato, u_aprox, N)
                err_max.append(err_N)
                Ni.append(N)
                N = N*2
            plot_graf2d(Ni, err_max, f"Erro por passos - Lambda = {lamb}")
        
        else:
            N = N_max
            delta_x = 1/N
            delta_t = (delta_x)**2 * lamb
            M = round(T/delta_t)
            u_aprox = sol_aprox_u(delta_x, delta_t, N, M, *inf['c'])
            plot_graf3d(u_aprox, T, N, M, lamb)
    
    elif teste > 3 and teste <= 9:                                      # Testes do EP1
        if not (teste == 6 or teste == 9):
            N = 10
            Ni = []
            err_max = []
            while (N <= N_max):
                delta_x = 1/N
                delta_t = delta_x
                M = N
                lamb = N
                if teste == 4 or teste == 7: caso = 'a'
                else: caso = 'b'
                if teste <= 5: metodo = euler
                else: metodo = crank
                u_aprox = metodo(delta_x, N, *inf[caso])
                u_exato = sol_exata_u(delta_x, delta_t, N, M, *inf[caso][:-1])
                if N == N_max:
                    plot_graf3d(u_aprox, T, N, M, lamb)
                err_N = err_u(u_exato, u_aprox, N)
                err_max.append(err_N)
                Ni.append(N)
                N = N*2
            plot_graf2d(Ni, err_max, f"Erro por passos - Lambda = {lamb}")

        else:
            N = N_max
            delta_x = 1/N_max
            delta_t = delta_x
            M = N_max
            lamb = N_max
            if teste == 6: metodo = euler
            else: metodo = crank
            u_aprox = metodo(delta_x, N, *inf['c'])
            plot_graf3d(u_aprox, T, N, M, lamb)

#    ----------------------------------------- INTERFACE DO EP2 COMEÇA AQUI ---------------------------------------------

    else:
        N = N_max                                                       # Variáveis definidas para Crank-Nicolson
        delta_x = 1/N_max
        delta_t = delta_x
        M = N_max
        lamb = N_max
        escolha = str(input('Caso (a, b, c, d) a ser rodado: ')).lower()    # Pergunta a escolha ao usuário
        ak, E2, u_kTxi_all, uT = implementa_ep2(escolha, delta_x, N, u0_c, g1_c, g2_c, f_p)    # RODA TODOS OS TESTES
        for i, a in enumerate(ak):                                          # Imprime os ak de forma organizada
            print(f'a{i+1}: {a}')
        print(f'Erro quadrático do teste: {E2}')                            # Imprime o erro quadrático
        plot_aprox = monta_plot(u_kTxi_all, uT)                             # Prepada o vetor aproximado para o plot
        plot_graf2d(np.linspace(0, 1, (len(plot_aprox))), np.array(plot_aprox),         # Plota o vetor calculado
                    'Resultado Aproximado', xlabel='Posição', ylabel='Temperatura')
        plot_graf2d(np.linspace(0, 1, (len(uT))), np.array(uT),                         # Plota o vetor real, dado
                    'Resultado Real', xlabel='Posição', ylabel='Temperatura')