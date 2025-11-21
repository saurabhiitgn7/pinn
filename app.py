import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_model import FCN
from physics import oscillator_solution

# --- CONFIGURATION ---
st.set_page_config(page_title="Interactive PINN Tutorial", layout="wide")

st.title("🧠 Physics-Informed Neural Networks (PINNs)")
st.markdown("""
This app demonstrates how a Neural Network can learn to solve a differential equation 
by using the **Laws of Physics** in its loss function, even with very little training data.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Problem Parameters")
d = st.sidebar.slider("Damping (d)", 0.0, 1.0, 0.2, 0.1)
w0 = st.sidebar.slider("Frequency (w0)", 2.0, 10.0, 4.0, 0.5)

st.sidebar.header("2. Training Parameters")
learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f")
iterations = st.sidebar.slider("Training Iterations", 100, 5000, 1000, 100)
physics_weight = st.sidebar.slider("Physics Loss Weight (Lambda)", 0.0, 10.0, 1.0, 0.1, 
                                   help="How much importance to give the ODE vs data points.")

st.sidebar.header("3. Data Availability")
n_data_points = st.sidebar.slider("Number of Training Data Points", 0, 20, 3, 
                                  help="How many 'answers' does the AI get to see?")

# --- MAIN CONTENT ---

# 1. Generate Ground Truth
t_physics = np.linspace(0, 1, 300).reshape(-1, 1) # Collocation points (dense)
exact_u = oscillator_solution(d, w0, t_physics)

# 2. Select sparse training data
indices = np.linspace(0, 299, n_data_points).astype(int)
t_data = t_physics[indices]
u_data = exact_u[indices]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("The Physics (ODE)")
    st.latex(r"m\frac{d^2u}{dt^2} + \mu\frac{du}{dt} + ku = 0")
    st.info(f"""
    We are trying to approximate the function $u(t)$.
    
    **Data Loss:** We give the NN {n_data_points} known points (Green Dots).
    
    **Physics Loss:** We tell the NN that for *every* point $t$, it must satisfy the equation above.
    """)
    
    start_btn = st.button("Train PINN Model", type="primary")

# --- TRAINING LOOP ---
if start_btn:
    # Convert to PyTorch Tensors
    t_physics_torch = torch.tensor(t_physics, dtype=torch.float32, requires_grad=True)
    t_data_torch = torch.tensor(t_data, dtype=torch.float32)
    u_data_torch = torch.tensor(u_data, dtype=torch.float32)

    # Initialize Model (Input=1 time, Hidden=3 layers of 32, Output=1 displacement)
    layers = [1, 32, 32, 32, 1]
    model = FCN(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Placeholders for live plotting
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    loss_history = []

    for i in range(iterations):
        optimizer.zero_grad()
        
        # 1. Loss based on Data (Standard NN training)
        loss1 = 0
        if n_data_points > 0:
            loss1 = model.loss_data(t_data_torch, u_data_torch)
        
        # 2. Loss based on Physics (PINN specific)
        loss2 = model.loss_physics(t_physics_torch, d, w0)
        
        # Total Loss
        loss = loss1 + (physics_weight * loss2)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if i % (iterations // 10) == 0:
            progress_bar.progress((i+1)/iterations)

    progress_bar.progress(100)

    # --- FINAL PLOT ---
    with col2:
        st.subheader("Results")
        
        # Get prediction
        model.eval()
        prediction = model(t_physics_torch).detach().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_physics, exact_u, 'k--', label="Exact Solution (Physics)", alpha=0.6)
        ax.plot(t_physics, prediction, 'r-', label="PINN Prediction", linewidth=2)
        ax.scatter(t_data, u_data, color='green', s=100, label="Training Data", zorder=5)
        
        ax.set_title(f"PINN Approximation (Points: {n_data_points}, Physics Weight: {physics_weight})")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Displacement (u)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.success(f"Final Loss: {loss.item():.6f}")