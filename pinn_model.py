import torch
import torch.nn as nn

class FCN(nn.Module):
    """Defines a standard Fully Connected Neural Network"""
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        # Initialize weights (Xavier)
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        
        # Normalize input to help training (simple min-max scaling logic could go here)
        a = x.float()
        
        for i in range(len(self.linears)-1):
            z = self.linears[i](a)
            a = self.activation(z)
        
        a = self.linears[-1](a)
        return a

    def loss_data(self, x_data, y_data):
        """Standard MSE Loss between prediction and actual data points"""
        y_pred = self.forward(x_data)
        return self.loss_function(y_pred, y_data)

    def loss_physics(self, x_physics, d, w0):
        """
        The PINN Magic. 
        We compute the derivatives of the network output with respect to input X.
        Then we check if those derivatives satisfy the ODE:
        u'' + 2d(u') + (w0^2)u = 0
        """
        # --- FIX IS HERE ---
        # We clone the input, DETACH it from any existing graph, 
        # and then enable gradient tracking. This ensures 'g' is a leaf variable.
        g = x_physics.clone().detach().requires_grad_(True)
        
        u = self.forward(g)
        
        # First derivative: u'
        u_t = torch.autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        
        # Second derivative: u''
        u_tt = torch.autograd.grad(u_t, g, torch.ones_like(u_t), create_graph=True)[0]
        
        # The Residual (How far off is the physics?)
        # ODE: u'' + 2d*u' + w0^2*u = 0
        physics_residual = u_tt + (2*d)*u_t + (w0**2)*u
        
        # We want this residual to be close to 0
        loss_f = torch.mean(physics_residual**2)
        return loss_f
