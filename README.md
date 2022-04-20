# Physics-Informed-Neural-Network

Physics Informed Neural Network ([PINN](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)) is a method to solve partial differential equations using neural networks developed by Maziar Raissi, Paris Perdikaris and George Em Karniadakis.  The method utilizes automatic differentiation in order to compute partial derivatives in the PDE.

The network serves as the function approximator, and after training will give the solution for the PDE.  The loss function is a combination of the MSE for the PDE given initial conditions and the MSE for the network itself.

I trained the network on Berger's equation over 25000 iterations and plotted the solution in space and time.  The network finished with a loss of 0.0011 after training according to the loss function.

In addition, I plotted the solution at every 100 iterations of training and made an animation of the solution converging over these iterations.  It is plotted in space from [-1,1] and time from [0,1].  The animation really gives intuition on how the model learns the solution.  The animation shows the model first learns the initial condition, then starts moving through time and finds the solution passed the initial condition.

Here is the animation!

![PINNConvergence](https://user-images.githubusercontent.com/67863882/164291260-53e2b874-b626-454c-94bd-14d58dbed7ae.gif)
