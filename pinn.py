import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm.auto import tqdm

#Setting the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#Create our network that will return u(t, x)

class Network(nn.Module):
    def __init__(self, num_layers, layer_size):
        super(Network, self).__init__()
    
        #2 inputs(x, t) so first layer has 2 inputs
        self.layers = [nn.Linear(2, layer_size), nn.Sigmoid()]

        for i in range(num_layers):
            self.layers += [nn.Linear(layer_size, layer_size), nn.Sigmoid()]
    
        #1 output
        self.layers += [nn.Linear(layer_size, 1)]

        self.network = nn.Sequential(*self.layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis = 1)
        return self.network(inputs)

#Initialize using Xavier weights

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

#Creating a model with 4 hidden layers and 100 nodes

model = Network(4, 100)
model.apply(init_weights)
model = model.to(device)

#Setting criterion and creating optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
model

#Now create the PDE which will be used as a loss function
#Utilize automatic differentiation for partial derivatives
#Here we are creating Burger's equation as used in original PINN paper
#f:=u_t+uu_x-(0.01/pi)u_xx

def f(x, t, model):
    u = model(x, t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph = True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph = True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph = True)[0]
    f = u_t + (u*u_x) - ((0.01/torch.pi)*u_xx)
    return f

#Must create some initial data
#Start with initial condition as initial guess
#Initial and Boundary Data:
#u(x,0) = -sin(pi*x)
#u(-1,t) = u(1,t) = 0

x_init = np.random.uniform(low = -1.0, high = 1.0, size = (3500,1))
t_init = np.zeros((3500,1))
t = np.random.uniform(low = 0.0, high = 1.0, size = (3500,1))
u_init = -np.sin(np.pi * x_init)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

#Setting variables that will be used for plotting

x_plot=np.arange(-1,1,0.01)
t_plot=np.arange(0,1,0.01)
x_mesh, t_mesh = np.meshgrid(x_plot, t_plot)
x_plot = np.ravel(x_mesh).reshape(-1,1)
t_plot = np.ravel(t_mesh).reshape(-1,1)

x_torch = Variable(torch.from_numpy(x_plot).float(), requires_grad=True).to(device)
t_torch = Variable(torch.from_numpy(t_plot).float(), requires_grad=True).to(device)

#Train our network on our data
epochs = 25000
losses = []
solutions = []
progress = tqdm(range(epochs))

for epoch in range(epochs):
    optimizer.zero_grad()
    #Loss function based on initial data
    x_in = Variable(torch.from_numpy(x_init).float(), requires_grad = False).to(device)
    t_in = Variable(torch.from_numpy(t_init).float(), requires_grad = False).to(device)
    u_in = Variable(torch.from_numpy(u_init).float(), requires_grad = False).to(device)

    u_out = model(x_in, t_in)
    MSE_u = criterion(u_out, u_in)

    #Loss function for PDE
    x_pde = Variable(torch.from_numpy(x_init).float(), requires_grad = True).to(device)
    t_pde = Variable(torch.from_numpy(t).float(), requires_grad = True).to(device)
    f_out = f(x_pde, t_pde, model)
 
    #Need tensor of zeros for MSE_f
    zeros = np.zeros((3500,1))
    zeros_ag = Variable(torch.from_numpy(zeros).float(), requires_grad = False).to(device)

    MSE_f = criterion(f_out, zeros_ag)

    #Combine loss
    loss = MSE_u + MSE_f

    #Save solution at every 100 epochs in order to plot animation
    if(epoch % 100 == 0):
        solutions.append(model(x_torch, t_torch))

    loss.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        losses.append(loss.data)
        
    progress.update(1)

#Plotting the solution after training
fig = plt.figure()
ax = fig.gca(projection='3d')

u_torch = model(x_torch,t_torch)
u=u_torch.data.cpu().numpy()
u_mesh = u.reshape(x_mesh.shape)

surf = ax.plot_surface(x_mesh,t_mesh,u_mesh, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
plt.close(fig)

#Detatching losses so it can be plotted
for i in range(len(losses)):
    losses[i] = losses[i].detach().cpu()

#Plotting Losses
x = np.linspace(1, epochs, epochs)
plt.plot(x, losses)
plt.title("Error rate over 25000 iterations")
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
plt.close(fig)

#Printing the final loss
print(losses[-1])

#Saving the solution after every 100 iterations in order to make animation
#of the solution converging over time

import os.path

save_path = '/home/jupyter/pinn_plots'

for i in range(1, len(solutions)):
    print(i)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    u_torch = solutions[i]
    u=u_torch.data.cpu().numpy()
    u_mesh = u.reshape(x_mesh.shape)

    surf = ax.plot_surface(x_mesh,t_mesh,u_mesh, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.patch.set_facecolor('white')
    #plt.show()
    file_name = str(i)
    complete_file_name = os.path.join(save_path, file_name+".png")
    plt.savefig(complete_file_name)
    plt.close(fig)

torch.save(model.state_dict(), 'Pinn.pt')
