import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pinn_model import InversePINN
from physics import oscillator_solution

st.set_page_config(page_title="AI Parameter Discovery", layout="wide")

st.title("🕵️ AI Physicist: Auto-Discovering Parameters")
st.markdown("""
**The Inverse Problem:** We give the AI noisy data from an experiment. 
It must simultaneously learn to fit the curve **AND** calculate the hidden physical constants ($d$ and $w_0$).
""")

# --- SIDEBAR: CREATE THE "REALITY" ---
st.sidebar.header("1. Secret Reality (Ground Truth)")
true_d = st.sidebar.slider("True Damping (d)", 0.0, 1.0, 0.3)
true_w0 = st.sidebar.slider("True Frequency (w0)", 2.0, 8.0, 5.0)

st.sidebar.header("2. Experiment Settings")
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.05, help="How messy is the data?")
n_points = st.sidebar.slider("Data Points", 10, 100, 40)

st.sidebar.header("3. AI Training")
lr = st.sidebar.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=1)

# --- UPDATED SLIDER (UP TO 50,000) ---
iterations = st.sidebar.slider("Iterations (Epochs)", min_value=1000, max_value=50000, value=10000, step=1000)

# --- MAIN APP ---

col1, col2 = st.columns([1, 1])

# 1. Generate "Experimental" Data
t_full = np.linspace(0, 1, 300).reshape(-1, 1)
u_clean = oscillator_solution(true_d, true_w0, t_full)

# Select random points for training data
idx = np.linspace(0, 299, n_points).astype(int)
t_train = t_full[idx]
# Add noise to simulate real world data
noise = np.random.normal(0, noise_level, u_clean[idx].shape)
u_train = u_clean[idx] + noise

# Display the challenge
with col1:
    st.subheader("The Challenge")
    fig_data, ax_d = plt.subplots(figsize=(6, 4))
    ax_d.plot(t_full, u_clean, 'k--', alpha=0.3, label="True Physics (Unknown to AI)")
    ax_d.scatter(t_train, u_train, c='red', label="Noisy Data (Given to AI)")
    ax_d.legend()
    st.pyplot(fig_data)
    
    st.info(f"The AI starts with a guess: d=0.0, w0=1.0")
    start = st.button("Start Discovery", type="primary")

if start:
    # Setup PyTorch Data
    t_phys_torch = torch.tensor(t_full, dtype=torch.float32)
    t_train_torch = torch.tensor(t_train, dtype=torch.float32)
    u_train_torch = torch.tensor(u_train, dtype=torch.float32)

    # Initialize Model
    model = InversePINN(layers=[1, 32, 32, 1])
    
    # We optimize Weights AND Parameters (d, w0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Tracking history for plots
    history = {'loss': [], 'd': [], 'w0': []}
    
    # Progress bars
    prog_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = col2.empty()
    
    # Calculate plot update interval (Update graph 10 times total to save speed)
    plot_interval = iterations // 10
    if plot_interval < 1: plot_interval = 1

    for i in range(iterations):
        optimizer.zero_grad()
        
        # Loss 1: Fit the noisy data
        loss_d = model.loss_data(t_train_torch, u_train_torch)
        
        # Loss 2: Obey Physics (with current d_param and w0_param)
        loss_p = model.loss_physics(t_phys_torch)
        
        loss = loss_d + loss_p
        loss.backward()
        optimizer.step()
        
        # Record stats
        curr_d, curr_w0 = model.get_params()
        history['loss'].append(loss.item())
        history['d'].append(curr_d)
        history['w0'].append(curr_w0)
        
        # Update Progress Bar logic
        if i % (iterations // 100) == 0:
            prog_bar.progress((i+1)/iterations)
            status_text.text(f"Iter {i}/{iterations}: Found d={curr_d:.4f}, w0={curr_w0:.4f}")
            
        # Live update plot (Only update 10 times total to prevent freezing)
        if i % plot_interval == 0 or i == iterations - 1:
            with chart_placeholder.container():
                # Plot 1: Parameter Convergence
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                
                ax1.plot(history['d'], label='Estimated d', color='blue')
                ax1.axhline(y=true_d, color='green', linestyle='--', label='True d', alpha=0.7)
                
                ax1.plot(history['w0'], label='Estimated w0', color='orange')
                ax1.axhline(y=true_w0, color='red', linestyle='--', label='True w0', alpha=0.7)
                
                ax1.set_title("Parameter Discovery Over Time")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Curve Fit
                model.eval()
                pred = model(t_phys_torch).detach().numpy()
                ax2.plot(t_full, u_clean, 'g--', alpha=0.5, label="True Clean")
                ax2.plot(t_full, pred, 'b-', label="AI Approximation")
                ax2.scatter(t_train, u_train, c='red', s=10, alpha=0.5)
                ax2.set_title(f"Current Physics: d={curr_d:.3f}, w0={curr_w0:.3f}")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close(fig)
                model.train()

    prog_bar.progress(100)
    
    # Final Result Display
    final_d, final_w0 = model.get_params()
    
    st.success("Discovery Complete!")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Damping (d)", f"{final_d:.4f}", delta=f"{final_d - true_d:.4f}")
    res_col2.metric("Frequency (w0)", f"{final_w0:.4f}", delta=f"{final_w0 - true_w0:.4f}")
    
    st.markdown("If the Delta (small arrow) is close to 0, the AI successfully discovered the hidden physics law from the noisy data.")
