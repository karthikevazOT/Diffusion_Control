from re import X
import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Parameters
N = 1000  # Number of samples
d = 2     # Number of states
Nt = 6000
dt = 0.0001
D = 2

x_traj = np.zeros((d, N, Nt))
x_traj[:, :, 0] = np.zeros((d, N))
x_traj[0, :, 0] = np.zeros((1, N))+0.75
x_traj[1, :, 0] = np.zeros((1, N))-0.75
# Define multiple circular obstacles: (center_x, center_y, radius)

def g_1(x, N):
    y = np.zeros((d,N), dtype=x.dtype)
    y[0, :] = 1  # x-direction component
    y[1,:] = 0  # y-direction component
    return y

def g_2(x, N):
    y = np.zeros((d,N), dtype=x.dtype)
    y[0, :] = 0  # x-direction component
    y[1,:] = 1  # y-direction component
    return y

obstacles = [
    (0.5, -0.3, 0.25),   # Obstacle 1 at (0.5, 0) with radius 0.2
    (-0.5, 0.0, 0.2),   # Obstacle 2 at (-0.5, 0) with radius 0.2
    (0, 0.5, 0.3)   # Obstacle 2 at (-0.5, 0) with radius 0.2
]

for it in range(Nt - 1):
    # SDE step
    x_new = x_traj[:, :, it] + ((D*dt)**0.5)*np.random.randn(1, N)*g_1(x_traj[:, :, it],N) + ((D*dt)**0.5)*np.random.randn(1, N)*g_2( x_traj[:, :, it],N)  
   

    # Reflect from circular obstacles
    for (cx, cy, r) in obstacles:
        # Compute displacement from obstacle center
        dx = x_new[0, :] - cx
        dy = x_new[1, :] - cy
        dist = np.sqrt(dx**2 + dy**2)

        # Find particles inside this obstacle
        inside_idx = dist < r
        if np.any(inside_idx):
            # Reflect these particles onto the obstacle boundary
            dx_norm = dx[inside_idx] / dist[inside_idx]
            dy_norm = dy[inside_idx] / dist[inside_idx]
            x_new[0, inside_idx] = cx + r * dx_norm
            x_new[1, inside_idx] = cy + r * dy_norm

    # Also reflect off the outer square domain
    x_new[0, :] = np.clip(x_new[0, :], -1, 1)
    x_new[1, :] = np.clip(x_new[1, :], -1, 1)

    # Update trajectory
    x_traj[:, :, it + 1] = x_new

# Plot final positions
plt.figure(figsize=(6,6))
plt.plot(x_traj[0, :, Nt-1], x_traj[1, :, Nt-1], 'o', markersize=2, alpha=0.5)

# Plot obstacles
theta = np.linspace(0, 2*np.pi, 100)
for (cx, cy, r) in obstacles:
    plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'r-')  # obstacle boundary

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Reflected Brownian Motion with Three Circular Obstacles')
plt.show()

#%%
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model for the scalar score function s(x, t; theta)
class ScoreNetworkScalar(nn.Module):
    def __init__(self, input_dim):
        super(ScoreNetworkScalar, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x_t):
        # Returns a shape (N * Nt, 1)
        return self.net(x_t)

# Parameters
d = 2      # Dimension of the state space
learning_rate = 0.0005
num_epochs = 35

# Assuming x_traj is the trajectory data with shape (d, N, Nt)
trajectories = torch.tensor(x_traj, dtype=torch.float32)  # Convert trajectory data to torch.Tensor

# Initialize two neural networks each outputting a scalar
model1 = ScoreNetworkScalar(input_dim=d + 1)
model2 = ScoreNetworkScalar(input_dim=d + 1)

optimizer1 = optim.SGD(list(model1.parameters()) + list(model2.parameters()), lr=0.1)

# Reshape the trajectories and vector fields to combine N and Nt
x_t_all = trajectories.permute(1, 2, 0).reshape(N * Nt, d)  # Shape (N * Nt, d)

# Create a time feature array of shape (N * Nt, 1) with normalized time steps or scaled by dt
time_steps = torch.arange(Nt, dtype=torch.float32).repeat(N).unsqueeze(1) * dt  # Shape (N * Nt, 1)

# Concatenate state and time into a single input array for each model
x_t_with_time_all = torch.cat((x_t_all, time_steps), dim=1)  # Shape (N * Nt, d + 1)

for epoch in range(num_epochs):
    x_t_with_time_all.requires_grad_(True)

    # Forward pass through both models
    s1 = model1(x_t_with_time_all).squeeze(-1)  # Shape (N * Nt,)
    s2 = model2(x_t_with_time_all).squeeze(-1)  # Shape (N * Nt,)
    
    

    # Compute squared terms for both s1 and s2
    squared_term = 0.5 * (s1**2 + s2**2)  # (N * Nt,)

    # Compute gradients with respect to x:
    s1_sum = s1.sum()
    grad_s1_all = torch.autograd.grad(s1_sum, x_t_with_time_all, create_graph=True)[0][:, :d]
    # grad_s1_all[:,0] is ds1/dx1, grad_s1_all[:,1] is ds1/dx2 if d=2

    s2_sum = s2.sum()
    grad_s2_all = torch.autograd.grad(s2_sum, x_t_with_time_all, create_graph=True)[0][:, :d]
    # grad_s2_all[:,0] is ds2/dx1, grad_s2_all[:,1] is ds2/dx2

    # If you previously summed ds1/dx1 + ds2/dx2, replicate that logic:
    # directional_derivative_all = ds1/dx1 + ds2/dx2, for example:
    directional_derivative_all = grad_s1_all[:, 0] + grad_s2_all[:, 1]

    total_loss = (squared_term + directional_derivative_all).mean()

    optimizer1.zero_grad()
    total_loss.backward()
    optimizer1.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item():.4f}")
#%%

import numpy as np
import torch
import matplotlib.pyplot as plt

# After training the model, we want to visualize the score at t=0.
model1.eval()  # Put the model in evaluation mode
model2.eval()  # Put the model in evaluation mode

# Define a grid in the domain
grid_size = 20
x_values = np.linspace(-1, 1, grid_size)
y_values = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x_values, y_values)

# Create input points (x, y, t=0)
points = np.stack([X.flatten(), Y.flatten()], axis=-1)  # Shape (grid_size^2, 2)
t = np.zeros((points.shape[0], 1), dtype=np.float32)+ 0.1  # t=0 for all points
points_t0 = np.concatenate([points, t], axis=1)         # Shape (grid_size^2, 3) since d=2 and we add time dim

# Convert to torch tensor
points_t0_tensor = torch.tensor(points_t0, dtype=torch.float32)

# Compute the score at t=0
with torch.no_grad():
    # The model should output a vector of shape (N,2) at each point
    U = model1(points_t0_tensor)  # shape: (grid_size^2, 2)
    V = model2(points_t0_tensor)  # shape: (grid_size^2, 2)




# Plot the quiver field
plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='r')
plt.title('Learned Score Function at t=0')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#%%

import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
d = 2             # dimension
D = 10           # diffusion coefficient used in forward model (adjust if different)

times = np.linspace(Nt*dt, 0, Nt)  # we go backward in time
times_torch = torch.tensor(times, dtype=torch.float32)

# Obstacles: list of (cx, cy, r)


# Boundary limits
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

# Suppose we have M test samples from the final distribution at t=1
M = 2000
# Example: start from a uniform or random distribution at t=1
np.random.seed(42)
X_final = np.random.uniform(low=-1, high=1, size=(M, d)).astype(np.float32)
X = torch.tensor(X_final, requires_grad=False)

# Assume model1 and model2 are already defined, trained, and set to eval mode
model1.eval()
model2.eval()

# Store trajectories for visualization (optional)
X_traj = np.zeros((M, d, Nt))
X_traj[:, :, 0] = X.detach().cpu().numpy()


def g1(x, N):
    y = torch.zeros((d,N), dtype=x.dtype)
    y[0, :] = 1  # x-direction component
    y[1,:] = 0  # y-direction component
    return y

def g2(x, N):
    y = torch.zeros((d,N), dtype=x.dtype)
    y[0, :] = 0  # x-direction component
    y[1,:] = 1  # y-direction component
    return y

with torch.no_grad():
    for i in range(Nt-1):
        t_current = times_torch[i]
        # Current states and time as input to the models
        t_input = t_current * torch.ones((M, 1), dtype=torch.float32)
        X_input = torch.cat([X, t_input], dim=1)  # shape: (M, d+1)

        # Compute scalar outputs from each model
        s1 = model1(X_input).squeeze(-1)  # (M,)
        s2 = model2(X_input).squeeze(-1)  # (M,)

        # Combine into a score vector
        # score shape: (M, 2)
        score = torch.stack([s1, s2], dim=1)

        g1u = g1(X.T,M).T
        g2u = g2(X.T,M).T
        
        u1 = D*(s1*g1u[:,0]+s2*g1u[:,1])
        u2 = D*(s1*g2u[:,0]+s2*g2u[:,1])

        # Update positions
        
        X[:,0] = X[:,0] +  dt*u1 
        X[:,1] = X[:,1] +  dt*u2 
        
     #   X = X + D*score * dt/2  # add noise if needed: X = X + score * dt + noise

        # Reflect from obstacles
        X_np = X.detach().cpu().numpy()
        for (cx, cy, r) in obstacles:
            dx = X_np[:, 0] - cx
            dy = X_np[:, 1] - cy
            dist = np.sqrt(dx**2 + dy**2)
            inside_idx = dist < r
            if np.any(inside_idx):
                dx_norm = dx[inside_idx] / dist[inside_idx]
                dy_norm = dy[inside_idx] / dist[inside_idx]
                X_np[inside_idx, 0] = cx + r * dx_norm
                X_np[inside_idx, 1] = cy + r * dy_norm

        # Reflect off the outer boundaries
        X_np[:, 0] = np.clip(X_np[:, 0], x_min, x_max)
        X_np[:, 1] = np.clip(X_np[:, 1], y_min, y_max)

        # Update X
        X = torch.tensor(X_np, dtype=torch.float32)

        # Store trajectory step
        X_traj[:, :, i+1] = X_np

# Now, plot final positions after reversing (should be at time t=0)
plt.figure(figsize=(6,6))
plt.scatter(X_traj[:, 0, -1], X_traj[:, 1, -1], s=2, alpha=0.5, label='Final Positions at t=0')

# Plot obstacles
theta = np.linspace(0, 2*np.pi, 100)
for (cx, cy, r) in obstacles:
    plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'r-')

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Reverse Process Sample Distribution at t=0')
plt.legend()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Suppose X_traj has shape (M, 2, Nt)
# M: number of particles, Nt: number of time steps

# Define obstacles
obstacles = [
    (0.5, -0.3, 0.25),   # Obstacle 1 at (0.5, 0) with radius 0.2
    (-0.5, 0.0, 0.2),   # Obstacle 2 at (-0.5, 0) with radius 0.2
    (0, 0.5, 0.3)   # Obstacle 2 at (-0.5, 0) with radius 0.2
]

# Plot boundaries
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

# --- Reduce number of frames ---
skip = 100  # Skip every 5 frames
max_frames = 2000
frame_indices = list(range(1, X_traj.shape[2], skip))[:max_frames]

# --- Setup figure ---
fig, ax = plt.subplots(figsize=(4, 4))  # Smaller figure
scat = ax.scatter(X_traj[:, 0, 0], X_traj[:, 1, 0], s=2, alpha=0.5)

# Plot obstacles
theta = np.linspace(0, 2 * np.pi, 100)
for (cx, cy, r) in obstacles:
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 'r-')

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_aspect('equal', adjustable='box')

# --- Update function ---
def update(frame_idx):
    frame = frame_indices[frame_idx]
    scat.set_offsets(X_traj[:, :, frame])
    ax.set_title(f'Denoising Reflected Brownian Motion with Obstacles')
    return scat,

# --- Create animation ---
anim = animation.FuncAnimation(
    fig, update,
    frames=len(frame_indices),
    interval=100,  # 10 FPS
    blit=True
)

# --- Save animation as GIF ---
anim.save('reverse_process_animation.gif', writer='imagemagick', fps=10, dpi=72)

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# We assume x_traj is already defined from the previous simulation
# x_traj shape: (d, N, Nt)
# d=2, N=1000, Nt=200 as in the given code

fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter(x_traj[0, :, 0], x_traj[1, :, 0], s=2, alpha=0.5)

# Plot obstacles
obstacles = [
    (0.5, -0.3, 0.25),   # Obstacle 1 at (0.5, 0) with radius 0.2
    (-0.5, 0.0, 0.2),   # Obstacle 2 at (-0.5, 0) with radius 0.2
    (0, 0.5, 0.3)   # Obstacle 2 at (-0.5, 0) with radius 0.2
]

theta = np.linspace(0, 2*np.pi, 100)
for (cx, cy, r) in obstacles:
    ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'r-')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_aspect('equal', adjustable='box')
ax.set_title('Reflected Brownian Motion with Three Circular Obstacles')

def update(frame):
    # frame is the index in time
    scat.set_offsets(np.c_[x_traj[0, :, frame], x_traj[1, :, frame]])
    return scat,

anim = animation.FuncAnimation(fig, update, frames=x_traj.shape[2], interval=100, blit=True)

anim.save('forward_process_animation.gif', writer='imagemagick', fps=200)

plt.show()
#%%

# Assuming x_traj is already defined (shape: [2, 1000, 200])

# --- Reduce number of frames ---
skip = 100  # Skip every 5 frames
max_frames = 2000  # Cap total frames to keep GIF size small
frame_indices = list(range(0, x_traj.shape[2], skip))[:max_frames]

# --- Setup figure ---
fig, ax = plt.subplots(figsize=(4, 4))  # Smaller figure for smaller file
scat = ax.scatter(x_traj[0, :, 0], x_traj[1, :, 0], s=2, alpha=0.5)

# Obstacles
obstacles = [
    (0.5, -0.3, 0.25),
    (-0.5, 0.0, 0.2),
    (0, 0.5, 0.3)
]

theta = np.linspace(0, 2 * np.pi, 100)
for (cx, cy, r) in obstacles:
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 'r-')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_aspect('equal', adjustable='box')
ax.set_title('Reflected Brownian Motion')

# --- Animation update ---
def update(frame_idx):
    frame = frame_indices[frame_idx]
    scat.set_offsets(np.c_[x_traj[0, :, frame], x_traj[1, :, frame]])
    return scat,

# --- Create animation ---
anim = animation.FuncAnimation(
    fig, update,
    frames=len(frame_indices),
    interval=100,  # 100 ms per frame = 10 FPS
    blit=True
)

# --- Save as GIF ---
anim.save('forward_process_animation.gif', writer='imagemagick', fps=10, dpi=72)

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde

# Parameters
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
grid_size = 200

# Grid for evaluating KDE
x, y = np.mgrid[x_min:x_max:grid_size*1j, y_min:y_max:grid_size*1j]
grid_coords = np.vstack([x.ravel(), y.ravel()])

# Prepare figure
fig, ax = plt.subplots(figsize=(4, 4))
density_img = ax.imshow(np.zeros_like(x), extent=[x_min, x_max, y_min, y_max],
                        cmap='viridis', origin='lower', alpha=0.8, animated=True)

# Obstacles (static)
obstacles = [
    (0.5, -0.3, 0.25),   # Obstacle 1 at (0.5, 0) with radius 0.2
    (-0.5, 0.0, 0.2),   # Obstacle 2 at (-0.5, 0) with radius 0.2
    (0, 0.5, 0.3)   # Obstacle 2 at (-0.5, 0) with radius 0.2
]


theta = np.linspace(0, 2 * np.pi, 100)
for (cx, cy, r) in obstacles:
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 'r-', linewidth=2)

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_aspect('equal')
ax.set_title('Density Evolution')

# Normalize density range for color consistency
all_positions = x_traj.transpose(2, 0, 1).reshape(-1, 2).T
global_kde = gaussian_kde(all_positions)
z_all = global_kde(grid_coords).reshape(x.shape)
vmin, vmax = 0, np.max(z_all) * 1.1

# Update function
def update(frame):
    positions = x_traj[:, :, frame].T  # shape (2, M)
    kde = gaussian_kde(positions)
    z = kde(grid_coords).reshape(x.shape)
    density_img.set_array(np.rot90(z))
    ax.set_title(f'Density at Frame {frame}/{X_traj.shape[2]-1}')
    return density_img,

# Animation
skip = 100
max_frames = 1000
frame_indices = list(range(0, x_traj.shape[2], skip))[:max_frames]

anim = animation.FuncAnimation(
    fig, update,
    frames=frame_indices,
    interval=100,
    blit=True
)

# Save to GIF
anim.save('density_evolution.gif', writer='imagemagick', fps=10, dpi=72)

plt.show()
