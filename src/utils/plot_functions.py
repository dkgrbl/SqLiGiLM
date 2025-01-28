

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
 

def plot_covariance_matrix_xxpp_log(cov_matrix, title=""):     
    M = cov_matrix.shape[0]

    X, Y = np.mgrid[0:M, 0:M]
    
    Cov_range = np.max([np.abs(np.min(cov_matrix)), np.max(cov_matrix)])

    fig, ax = plt.subplots( )
    ax.set_title(title)
    pcm = ax.pcolormesh(X, Y, cov_matrix,
                       norm=colors.SymLogNorm(linthresh=np.min(np.diag(cov_matrix)) * 1e-1, linscale=1,
                                              vmin=-Cov_range, vmax=Cov_range, base=10), rasterized=True, cmap="RdBu_r", shading='auto')
    fig.colorbar(pcm, ax=ax)
    ax.invert_yaxis()    
    ax.axvline(M/2-0.5, color="k", ls="--")
    ax.axhline(M/2-0.5, color="k", ls="--")
    ax.set_xticks([0, M/2], labels= [r"$Q_0$", r"$P_0$"] )   
    ax.set_yticks([0, M/2], labels= [r"$Q_0$", r"$P_0$"] )    
    
    ax.set_aspect('equal')
    ax.xaxis.tick_top()

    plt.tight_layout()
 
def plot_squeezing(Covariance_matrix):
    squeezing = np.real(np.diag(Covariance_matrix))
    M = int(squeezing.size/2)
 
    plt.figure()
    plt.plot(np.linspace(0, M-1, M), 10.*np.log10(squeezing[0:M]),  color="red",  marker="o", label=r"$(\Delta Q)^2$",)
    plt.plot(np.linspace(0, M-1, M), 10.*np.log10(squeezing[M:2*M]), color="blue", marker="o", label=r"$(\Delta P)^2 $")
    plt.legend()
    plt.ylabel("Quadrature variance, dB")
    plt.xlabel("Mode number, dB")   

    plt.tight_layout()

def plot_mode_profile(U_matrix, n=0):
    if np.abs(n) > U_matrix.shape[0]:
        raise ValueError("N is out of range in U_matrix")  


    fig, axs = plt.subplots(2, 2, layout='constrained')

    ax = axs[0][0]
    ax.plot( np.abs(U_matrix[n,:]) )
    ax.set_title('Mode {}, abs'.format(n))
 
    ax = axs[0][1]
    ax.plot( np.angle(U_matrix[n,:]) )
    ax.set_title('Mode {}, phase'.format(n))

    ax = axs[1][0]
    ax.plot( np.real(U_matrix[n,:]) )
    ax.set_title('Mode {}, real'.format(n))

    ax = axs[1][1]
    ax.plot( np.imag(U_matrix[n,:]) )
    ax.set_title('Mode {}, imag'.format(n))
    
  
def plot_2d(X, Y, matrix_2d , title="", xlabel="", ylabel=""):
    x_mesh, y_mesh = np.meshgrid(X, Y)
    plt.figure()

    plt.title(title)
    plt.pcolormesh(x_mesh, y_mesh, matrix_2d, cmap='viridis',
               vmin=np.min(matrix_2d), vmax=np.max(matrix_2d), rasterized=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.colorbar()
    plt.tight_layout()


def plot_matrix(matrix, value="abs", title=""):
    plt.figure()
    plt.title(title + ",   " +  value)
    if value == "abs":
        plt.imshow(np.abs(matrix), cmap='viridis',)
    if value == "angle" or value == "phase":
        plt.imshow(np.angle(matrix), cmap='twilight',)
    if value == "real":
        plt.imshow(np.real(matrix), cmap='viridis',)
    if value == "imag":
        plt.imshow(np.imag(matrix), cmap='viridis',)
    plt.colorbar()