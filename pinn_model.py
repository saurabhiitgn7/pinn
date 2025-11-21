import torch
import torch.nn as nn

class InversePINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Standard Neural Network Layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        # Initialize weights
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

        # --- LEARNABLE PHYSICS PARAMETERS ---
        # We initialize them with a wrong guess (e.g., d=0, w0=1)
        # The AI must learn the true values (e.g., d=0.2, w0=4.0) during training.
        self.d_param = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.w0_param = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.linears)-1):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss_data(self, x_data, y_data):
        y_pred = self.forward(x_data)
        return self.loss_function(y_pred, y_data)

    def loss_physics(self, x_physics):
        """
        Here we use self.d_param and self.w0_param instead of passing them in.
        The optimizer will adjust these parameters to minimize this residual.
        """
        g = x_physics.clone().detach().requires_grad_(True)
        u = self.forward(g)
        
        u_t = torch.autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, g, torch.ones_like(u_t), create_graph=True)[0]
        
        # ODE using the LEARNABLE parameters
        physics_residual = u_tt + (2 * self.d_param) * u_t + (self.w0_param**2) * u
        
        return torch.mean(physics_residual**2)
    
    def get_params(self):
        """Helper to extract current parameter estimates"""
        return self.d_param.item(), self.w0_param.item()
