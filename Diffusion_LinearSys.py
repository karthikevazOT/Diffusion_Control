import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.linalg import norm
import scipy.linalg as la
import imageio
import os

def controllability_gramian(d, T):
    """
    Computes the controllability Gramian for a damped double integrator over [0, T].
    
    Parameters:
    d : float - Damping coefficient
    T : float - Time horizon

    Returns:
    Wc : (2,2) ndarray - Controllability Gramian matrix
    """
    if d == 0:
        raise ValueError("Damping coefficient must be nonzero to avoid singularities.")

    # Compute Gramian entries
    W11 = (T / d**2) - (3 / (2 * d**3)) + (2 * np.exp(-T * d) / d**3) - (np.exp(-2 * T * d) / (2 * d**3))
    W12 = ((-np.exp(2 * T * d) + 2 * np.exp(T * d) - 1) * np.exp(-2 * T * d)) / (2 * d**2)
    W22 = (1 / (2 * d)) - (np.exp(-2 * T * d) / (2 * d))

    # Construct Gramian matrix
    Wc = np.array([[W11, W12], [W12, W22]])

    return Wc

# Example usage
d = 1.0   # Damping coefficient
T = 5.0   # Time horizon
Wc = controllability_gramian(d, T)



def matrix_exponential(d, tau):
    """
    Computes the matrix exponential e^(A*tau) for the damped double integrator.

    Parameters:
    d : float - Damping coefficient
    tau : float - Time step

    Returns:
    eAt : (2,2) ndarray - Matrix exponential e^(A*tau)
    """
    # Define system matrix A

    
    # Compute matrix exponential
    eAt = expm(-A * tau)

    return eAt

# Example usage
d = 1.0   # Damping coefficient
tau = 2.0  # Time step

A = np.array([[0, 1],
              [0, d]])   # A matrix for reverse process; Must be unstable

B = np.array([0,1])

eAt = matrix_exponential(d, tau)

T = 10 # Final time

x1 = np.array([5,0])
x2 = np.array([-5,0])


def numerical_controllability_gramian(A, B, T, Nt=100):
    """
    Compute the finite-horizon controllability Gramian numerically.

    Parameters:
        A (ndarray): System matrix (n x n)
        B (ndarray): Input matrix (n x m)
        T (float): Time horizon
        Nt (int): Number of discretization steps

    Returns:
        Wc (ndarray): Controllability Gramian (n x n)
    """
    n = A.shape[0]
    dt = T / Nt
    time_steps = np.linspace(0, T, Nt)
    Wc = np.zeros((n, n))
    B = B.reshape(-1, 1)
    for t in time_steps:
        eAt = la.expm(-A * t)
        term = eAt @ B @ B.T @ eAt.T
        Wc += term * dt

    return Wc




def gradkernel(x,y,t):
    
    #W_t = numerical_controllability_gramian(d, t) 
    W_t = controllability_gramian(d, t) 
    
    #W_t= numerical_controllability_gramian(-A, B, t)
    Qtinv = np.linalg.inv(W_t)
    detQt = np.linalg.det(W_t)
    
    xponent1 = y-np.dot(matrix_exponential(d,t),x0) 
    powr = -1/2*np.dot(xponent1,np.dot(Qtinv,xponent1))
    
    K = 1/(2*np.pi)/detQt**(1/2)*np.exp(powr)
    
    gradK = -np.dot(Qtinv,xponent1)
    
    return gradK


def gradkernel2(x1,x2,y,t):
    
    
    #W_t = numerical_controllability_gramian(A, B, T)
    W_t = controllability_gramian(d, t) 
    


    #W_t= numerical_controllability_gramian(-A, B, t)
    Qtinv = np.linalg.inv(W_t)
    detQt = np.linalg.det(W_t)
    
    xponent1 = y-np.dot(matrix_exponential(d,t),x1)
    xponent2 = y-np.dot(matrix_exponential(d,t),x2) 
    powr1 = -1/2*np.dot(xponent1,np.dot(Qtinv,xponent1))
    powr2 = -1/2*np.dot(xponent2,np.dot(Qtinv,xponent2))
    
    K1 = 1/(2*np.pi)/detQt**(1/2)*np.exp(powr1)
    K2 = 1/(2*np.pi)/detQt**(1/2)*np.exp(powr2)
    
    gradK1 = -K1*np.dot(Qtinv,xponent1)
    gradK2 = -K2*np.dot(Qtinv,xponent2)
    
    score = (0.5*gradK1+0.5*gradK2)/(0.5*K1+0.5*K2)

    return score



def ode_system(t,y):
    
    
    u = gradkernel2(x1,x2,y,T-t+0.0000001)[1]
    
    dydt = A @ y + B * u
    

    return dydt

timrange = np.arange(0, T, 0.001)

Np=200
yall = np.zeros((Np,2,np.size(timrange)))

for N in range(Np):
    
 yinit = 3*np.random.normal((0,1))

 sol = solve_ivp(ode_system, [0, T], yinit, t_eval=timrange)
    
 yall[N,:,:] = sol.y    
 
    
#%%
    
for i in range(np.size(timrange)):
    
    if (i % 100 == 0):
      plt.plot(yall[:,0,i],yall[:,1,i],'o')
      plt.xlim([-10,10])
      plt.ylim([-10,10])
      plt.pause(0.1)
      

plt.plot(yall[:,0,0],yall[:,1,0],'o')
plt.plot(yall[:,0,-1],yall[:,1,-1],'o')

plt.xlim([-10,10])
plt.ylim([-10,10])

plt.pause(0.1)      
      
  #%%    
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import imageio
import os

# Parameters
frame_dir = "frames"
gif_filename = "trajectories.gif"
fps = 10  # frames per second

# Ensure frame directory exists
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

# Clean up any old frames
for f in os.listdir(frame_dir):
    os.remove(os.path.join(frame_dir, f))

# Save frames
frames = []
for i in range(np.size(timrange)):
    if i % 100 == 0:  # Save frame every 100 time steps
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")

        plt.figure(figsize=(6, 6))
        plt.plot(yall[:, 0, i], yall[:, 1, i], 'o', markersize=3)
        plt.plot(yall[:,0,-1],yall[:,1,-1],'o')
        plt.legend(['Current','Final'])
        plt.title(f"Time: {timrange[i]:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
  
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

# Save GIF
imageio.mimsave(gif_filename, frames, fps=fps)
print(f"GIF saved as {gif_filename}")
