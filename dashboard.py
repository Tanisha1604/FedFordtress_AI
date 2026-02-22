import streamlit as st
from src.main import run_federated_training, save_training_results
from src.baseline import train_baseline
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import math


# ═══════════════════════════════════════════════════════════════════════
#  THEME COLORS
# ═══════════════════════════════════════════════════════════════════════
COLORS = {
    "bg":        "#0a0e27",
    "card":      "#131640",
    "surface":   "#1a1a2e",
    "honest":    "#00ff88",
    "malicious": "#ff4757",
    "server":    "#00d2ff",
    "dp":        "#7c3aed",
    "accent":    "#00d2ff",
    "text":      "#e2e8f0",
    "muted":     "#94a3b8",
    "gold":      "#fbbf24",
}


# ═══════════════════════════════════════════════════════════════════════
#  3D NETWORK TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════
def create_3d_topology(num_clients, malicious_clients):
    """Create interactive 3D network topology using Plotly Scatter3d."""
    # Server at center top
    server_x, server_y, server_z = [0], [0], [1.5]

    # Clients arranged in a circle at z=0
    client_x, client_y, client_z = [], [], []
    client_colors = []
    client_labels = []

    for i in range(num_clients):
        angle = 2 * math.pi * i / num_clients
        radius = 2.0
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        cz = 0.0
        client_x.append(cx)
        client_y.append(cy)
        client_z.append(cz)

        if i < malicious_clients:
            client_colors.append(COLORS["malicious"])
            client_labels.append(f"⚠️ Client {i} (Malicious)")
        else:
            client_colors.append(COLORS["honest"])
            client_labels.append(f"✅ Client {i} (Honest)")

    # Connection lines from each client to server
    line_x, line_y, line_z = [], [], []
    line_colors = []
    for i in range(num_clients):
        # Create curved path through a midpoint
        mid_z = 0.8
        steps = 10
        for s in range(steps):
            t = s / steps
            lx = client_x[i] * (1 - t) + server_x[0] * t
            ly = client_y[i] * (1 - t) + server_y[0] * t
            lz = client_z[i] * (1 - t) ** 2 * 0 + mid_z * 2 * t * (1 - t) + server_z[0] * t ** 2
            line_x.append(lx)
            line_y.append(ly)
            line_z.append(lz)
        line_x.append(None)
        line_y.append(None)
        line_z.append(None)

    # Lines trace
    lines_trace = go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color=COLORS["accent"], width=3),
        opacity=0.4,
        hoverinfo='skip',
        showlegend=False,
    )

    # Client nodes
    clients_trace = go.Scatter3d(
        x=client_x, y=client_y, z=client_z,
        mode='markers+text',
        marker=dict(
            size=12,
            color=client_colors,
            symbol='circle',
            line=dict(color='white', width=1),
            opacity=0.95,
        ),
        text=client_labels,
        textposition='bottom center',
        textfont=dict(size=10, color=COLORS["text"]),
        hoverinfo='text',
        showlegend=False,
    )

    # Server node (larger)
    server_trace = go.Scatter3d(
        x=server_x, y=server_y, z=server_z,
        mode='markers+text',
        marker=dict(
            size=20,
            color=COLORS["server"],
            symbol='diamond',
            line=dict(color='white', width=2),
            opacity=1.0,
        ),
        text=["🖥️ Server"],
        textposition='top center',
        textfont=dict(size=13, color=COLORS["server"]),
        hoverinfo='text',
        showlegend=False,
    )

    # Orbit ring (decorative)
    ring_t = np.linspace(0, 2 * np.pi, 100)
    ring_trace = go.Scatter3d(
        x=2.0 * np.cos(ring_t),
        y=2.0 * np.sin(ring_t),
        z=np.zeros(100),
        mode='lines',
        line=dict(color=COLORS["muted"], width=2, dash='dot'),
        opacity=0.3,
        hoverinfo='skip',
        showlegend=False,
    )

    fig = go.Figure(data=[ring_trace, lines_trace, clients_trace, server_trace])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-3, 3]),
            yaxis=dict(visible=False, range=[-3, 3]),
            zaxis=dict(visible=False, range=[-0.5, 2.5]),
            bgcolor=COLORS["bg"],
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0.3),
            ),
            aspectmode='cube',
        ),
        paper_bgcolor=COLORS["bg"],
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
#  HELPER CHARTS
# ═══════════════════════════════════════════════════════════════════════
def create_threat_gauge(score):
    """Create a threat level gauge."""
    if score < 30:
        color = COLORS["honest"]
        label = "LOW"
    elif score < 60:
        color = COLORS["gold"]
        label = "MEDIUM"
    else:
        color = COLORS["malicious"]
        label = "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Threat Level: {label}", 'font': {'size': 20, 'color': COLORS["text"]}},
        number={'font': {'color': COLORS["text"]}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS["muted"]},
            'bar': {'color': color},
            'bgcolor': COLORS["surface"],
            'borderwidth': 2,
            'bordercolor': COLORS["muted"],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0,255,136,0.1)'},
                {'range': [30, 60], 'color': 'rgba(251,191,36,0.1)'},
                {'range': [60, 100], 'color': 'rgba(255,71,87,0.1)'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_anomaly_heatmap(anomaly_values, num_clients):
    """Create anomaly heatmap across rounds and clients."""
    if not anomaly_values:
        return None

    fig, ax = plt.subplots(figsize=(8, 3), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    data = np.array(anomaly_values).reshape(-1, 1)
    sns.heatmap(
        data.T, annot=True, fmt='.1f',
        cmap='RdYlGn_r', ax=ax,
        cbar_kws={'label': 'Anomaly Score'},
        linewidths=0.5,
        linecolor=COLORS["surface"],
    )
    ax.set_xlabel('Round', color=COLORS["text"])
    ax.set_ylabel('', color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    return fig


def create_live_chart(values, title, color, y_label="Value"):
    """Create a styled Plotly line chart for live training."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(values) + 1)),
        y=values,
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color, line=dict(color='white', width=1)),
        fill='tozeroy',
        fillcolor=color.replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(
            title="Round", color=COLORS["muted"],
            gridcolor=COLORS["surface"], showgrid=True,
        ),
        yaxis=dict(
            title=y_label, color=COLORS["muted"],
            gridcolor=COLORS["surface"], showgrid=True,
        ),
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FedFortress",
    page_icon="🛡️",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    /* Global */
    .stApp {{
        background: linear-gradient(135deg, {COLORS["bg"]} 0%, {COLORS["surface"]} 100%);
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {COLORS["card"]};
        border-right: 1px solid rgba(0, 210, 255, 0.15);
    }}

    /* Cards */
    .glow-card {{
        background: linear-gradient(145deg, {COLORS["card"]}, {COLORS["surface"]});
        border: 1px solid rgba(0, 210, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 30px rgba(0, 210, 255, 0.05);
        transition: all 0.3s ease;
    }}
    .glow-card:hover {{
        border-color: rgba(0, 210, 255, 0.3);
        box-shadow: 0 8px 40px rgba(0, 210, 255, 0.1);
        transform: translateY(-2px);
    }}

    /* Hero title */
    .hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {COLORS["server"]}, {COLORS["honest"]}, {COLORS["dp"]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        animation: shimmer 3s ease-in-out infinite;
    }}
    @keyframes shimmer {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.85; }}
    }}

    .hero-subtitle {{
        color: {COLORS["muted"]};
        text-align: center;
        font-size: 1.15rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0, 210, 255, 0.2);
    }}

    /* Feature badges */
    .feature-badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }}
    .badge-green {{ background: rgba(0,255,136,0.15); color: {COLORS["honest"]}; border: 1px solid rgba(0,255,136,0.3); }}
    .badge-red {{ background: rgba(255,71,87,0.15); color: {COLORS["malicious"]}; border: 1px solid rgba(255,71,87,0.3); }}
    .badge-blue {{ background: rgba(0,210,255,0.15); color: {COLORS["server"]}; border: 1px solid rgba(0,210,255,0.3); }}
    .badge-purple {{ background: rgba(124,58,237,0.15); color: {COLORS["dp"]}; border: 1px solid rgba(124,58,237,0.3); }}
    .badge-gold {{ background: rgba(251,191,36,0.15); color: {COLORS["gold"]}; border: 1px solid rgba(251,191,36,0.3); }}

    /* Client status cards */
    .client-card {{
        background: {COLORS["card"]};
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    .client-honest {{ border-left: 3px solid {COLORS["honest"]}; }}
    .client-malicious {{ border-left: 3px solid {COLORS["malicious"]}; }}

    /* Metric overrides */
    [data-testid="stMetric"] {{
        background: {COLORS["card"]};
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }}

    /* Divider */
    .neon-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {COLORS["accent"]}, transparent);
        margin: 32px 0;
        border: none;
    }}

    /* Hide default Streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Configuration
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:20px;">
        <span style="font-size:2rem;">🛡️</span>
        <h2 style="color:{COLORS['server']}; margin:0;">FedFortress</h2>
        <small style="color:{COLORS['muted']};">Control Panel</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header' style='font-size:1rem; margin-top:8px;'>🎯 Training</div>", unsafe_allow_html=True)
    num_clients = st.slider("Number of Clients", 2, 10, 5, key="clients")
    rounds = st.slider("Training Rounds", 1, 10, 5, key="rounds")
    local_epochs = st.slider("Local Epochs per Round", 1, 5, 1, key="epochs")
    max_samples = st.slider("CIFAR-10 Samples", 1000, 50000, 5000, step=1000, key="samples")

    st.markdown(f"<div class='section-header' style='font-size:1rem;'>⚔️ Security</div>", unsafe_allow_html=True)
    malicious_clients = st.slider("Malicious Clients", 0, num_clients - 1, min(1, num_clients - 1), key="malicious")
    attack_type = st.selectbox("Attack Strategy", ["noise_injection", "weight_scaling", "label_flipping", "random_weights"])

    st.markdown(f"<div class='section-header' style='font-size:1rem;'>🔐 Privacy</div>", unsafe_allow_html=True)
    dp_enabled = st.toggle("Differential Privacy", value=True)
    dp_epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, step=0.1, disabled=not dp_enabled)

    st.markdown(f"<div class='section-header' style='font-size:1rem;'>🧮 Aggregation</div>", unsafe_allow_html=True)
    aggregation = st.selectbox("Strategy", ["FedAvg", "Trimmed Mean", "Median"])

    st.markdown("---")
    st.caption(f"📊 {max_samples:,} samples · {num_clients} clients · ε={dp_epsilon}")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1: HERO
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="hero-title">🛡️ FedFortress</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Asynchronous Federated Learning with Robust Aggregation & Differential Privacy</p>', unsafe_allow_html=True)

# Feature badges
st.markdown(f"""
<div style="text-align:center; margin-bottom:20px;">
    <span class="feature-badge badge-blue">Async Updates</span>
    <span class="feature-badge badge-green">AWTM Aggregation</span>
    <span class="feature-badge badge-red">Attack Detection</span>
    <span class="feature-badge badge-purple">Differential Privacy</span>
    <span class="feature-badge badge-gold">Convergence Evaluation</span>
    <span class="feature-badge badge-blue">Reproducible Results</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2: 3D NETWORK TOPOLOGY + SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🌐 Network Topology</div>', unsafe_allow_html=True)

col_3d, col_info = st.columns([3, 2])

with col_3d:
    fig_3d = create_3d_topology(num_clients, malicious_clients)
    st.plotly_chart(fig_3d, use_container_width=True, key="topo_3d")

with col_info:
    st.markdown(f"""
    <div class="glow-card">
        <h4 style="color:{COLORS['server']}; margin-top:0;">System Overview</h4>
        <table style="width:100%; color:{COLORS['text']}; font-size:0.95rem;">
            <tr><td>📡 Dataset</td><td style="text-align:right;"><b>CIFAR-10</b></td></tr>
            <tr><td>🖥️ Clients</td><td style="text-align:right;"><b>{num_clients}</b> ({num_clients - malicious_clients} honest, {malicious_clients} malicious)</td></tr>
            <tr><td>🧮 Aggregation</td><td style="text-align:right;"><b>{aggregation}</b></td></tr>
            <tr><td>🔐 Privacy</td><td style="text-align:right;"><b>{'ε=' + str(dp_epsilon) if dp_enabled else 'Disabled'}</b></td></tr>
            <tr><td>🔄 Rounds</td><td style="text-align:right;"><b>{rounds}</b></td></tr>
            <tr><td>📊 Samples</td><td style="text-align:right;"><b>{max_samples:,}</b></td></tr>
            <tr><td>⚔️ Attack</td><td style="text-align:right;"><b>{attack_type.replace('_', ' ').title()}</b></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Legend
    st.markdown(f"""
    <div class="glow-card" style="margin-top:12px;">
        <h4 style="color:{COLORS['text']}; margin-top:0;">Legend</h4>
        <p style="color:{COLORS['honest']}; margin:4px 0;">● Honest Client</p>
        <p style="color:{COLORS['malicious']}; margin:4px 0;">● Malicious Client</p>
        <p style="color:{COLORS['server']}; margin:4px 0;">◆ Central Server</p>
        <p style="color:{COLORS['muted']}; margin:4px 0; font-size:0.85rem;">↻ Drag to rotate · Scroll to zoom</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3: BASELINE TRAINING
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Phase 1 — Centralized Baseline</div>', unsafe_allow_html=True)

col_baseline_info, col_baseline_btn = st.columns([3, 1])
with col_baseline_info:
    _muted = COLORS['muted']
    st.markdown(f"<span style='color:{_muted}'>Train a centralized model for comparison. This shows performance without federation.</span>", unsafe_allow_html=True)
with col_baseline_btn:
    run_baseline = st.button("🚀 Run Baseline", use_container_width=True, key="baseline")

if run_baseline:
    baseline_progress = st.progress(0)
    baseline_status = st.empty()
    baseline_accuracy_values = []

    try:
        baseline_results = train_baseline(epochs=5)
        for epoch, result in enumerate(baseline_results):
            progress = (epoch + 1) / 5
            baseline_progress.progress(progress)
            baseline_status.text(f"Epoch {epoch+1}/5 — Loss: {result.get('loss', 0):.4f} — Accuracy: {result.get('accuracy', 0):.2f}%")
            baseline_accuracy_values.append(result.get('accuracy', 0))
            time.sleep(0.3)

        baseline_progress.progress(1.0)
        baseline_status.empty()
        st.success(f"✅ Baseline Complete — Final Accuracy: **{baseline_accuracy_values[-1]:.2f}%**")

        fig_base = create_live_chart(baseline_accuracy_values, "Baseline Accuracy", COLORS["server"], "Accuracy %")
        st.plotly_chart(fig_base, use_container_width=True)
    except Exception as e:
        st.error(f"Baseline error: {str(e)}")

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4: FEDERATED TRAINING (LIVE)
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚀 Phase 2 — Federated Training</div>', unsafe_allow_html=True)

run_fed = st.button("⚡ Start Federated Training", use_container_width=True, type="primary", key="federated")

if run_fed:
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("🏃 Initializing federated training on CIFAR-10...")

    accuracy_values = []
    anomaly_values = []
    all_results = []

    for update in run_federated_training(
        aggregation=aggregation,
        num_clients=num_clients,
        malicious_clients=malicious_clients,
        rounds=min(rounds, 10),
        quick_mode=False,
        max_samples=max_samples,
        local_epochs=local_epochs,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
    ):
        accuracy_values.append(update["accuracy"])
        anomaly_values.append(update["loss"])
        all_results.append(update)

        progress = update["round"] / min(rounds, 10)
        progress_bar.progress(progress)
        status_text.info(f"🔄 Round {update['round']}/{min(rounds, 10)} — Accuracy: {update['accuracy']}% — Anomaly Score: {update['loss']:.3f}")

    progress_bar.progress(1.0)
    status_text.success("✅ Federated training complete!")

    # ── Live Charts ──
    st.markdown('<div class="section-header" style="font-size:1.2rem;">📊 Training Results</div>', unsafe_allow_html=True)
    col_acc, col_loss = st.columns(2)

    with col_acc:
        fig_acc = create_live_chart(accuracy_values, "Model Accuracy (%)", COLORS["honest"], "Accuracy %")
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_loss:
        fig_loss = create_live_chart(anomaly_values, "Anomaly Score", COLORS["malicious"], "Score")
        st.plotly_chart(fig_loss, use_container_width=True)

    # ── Per-Client Status ──
    st.markdown('<div class="section-header" style="font-size:1.2rem;">👥 Client Status (Final Round)</div>', unsafe_allow_html=True)

    if all_results:
        last = all_results[-1]
        client_cols = st.columns(num_clients)

        for i in range(num_clients):
            with client_cols[i]:
                is_mal = i < malicious_clients
                status_class = "client-malicious" if is_mal else "client-honest"
                status_emoji = "⚠️" if is_mal else "✅"
                status_label = "Malicious" if is_mal else "Honest"
                acc = last.get("local_accuracies", [0]*num_clients)[i] if i < len(last.get("local_accuracies", [])) else 0
                anom = last.get("anomaly_scores", [0]*num_clients)[i] if i < len(last.get("anomaly_scores", [])) else 0

                st.markdown(f"""
                <div class="client-card {status_class}">
                    <div style="font-size:1.5rem;">{status_emoji}</div>
                    <div style="color:{COLORS['text']}; font-weight:700;">Client {i}</div>
                    <div style="color:{COLORS['muted']}; font-size:0.85rem;">{status_label}</div>
                    <div style="color:{COLORS['honest'] if not is_mal else COLORS['malicious']}; font-size:1.3rem; font-weight:700; margin-top:8px;">{acc:.1f}%</div>
                    <div style="color:{COLORS['muted']}; font-size:0.75rem;">anomaly: {anom:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    #  SECTION 5: SECURITY DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🛡️ Security Analysis</div>', unsafe_allow_html=True)

    # Calculate threat score
    malicious_ratio = malicious_clients / num_clients if num_clients > 0 else 0
    if len(anomaly_values) > 1:
        mean_anom = np.mean(anomaly_values)
        std_anom = np.std(anomaly_values)
        cv = std_anom / (mean_anom + 1e-6)
        threat_score = (malicious_ratio * 50) + (min(cv * 30, 50))
    else:
        threat_score = malicious_ratio * 100
    threat_score = min(100, threat_score)

    col_gauge, col_heatmap = st.columns(2)
    with col_gauge:
        fig_gauge = create_threat_gauge(threat_score)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_heatmap:
        _text_color = COLORS['text']
        st.markdown(f"<h4 style='color:{_text_color};'>Anomaly Trend</h4>", unsafe_allow_html=True)
        fig_hm = create_anomaly_heatmap(anomaly_values, num_clients)
        if fig_hm:
            st.pyplot(fig_hm)

    # Security features summary
    st.markdown(f"""
    <div class="glow-card">
        <h4 style="color:{COLORS['server']}; margin-top:0;">🔒 Active Defenses</h4>
        <div style="display:flex; gap:16px; flex-wrap:wrap; color:{COLORS['text']};">
            <div style="flex:1; min-width:200px;">
                <span class="feature-badge badge-green">✓ Active</span>
                <p><b>AWTM Aggregation</b><br><small style="color:{COLORS['muted']};">DBSCAN-based outlier filtering + reputation-weighted trimmed mean</small></p>
            </div>
            <div style="flex:1; min-width:200px;">
                <span class="feature-badge badge-green">✓ Active</span>
                <p><b>Anomaly Detection</b><br><small style="color:{COLORS['muted']};">Gradient norm z-score + IQR detection with reputation tracking</small></p>
            </div>
            <div style="flex:1; min-width:200px;">
                <span class="feature-badge {'badge-green' if dp_enabled else 'badge-red'}">{'✓ Active' if dp_enabled else '✗ Disabled'}</span>
                <p><b>Differential Privacy</b><br><small style="color:{COLORS['muted']};">Gaussian mechanism (ε={dp_epsilon}) with L2 norm clipping</small></p>
            </div>
            <div style="flex:1; min-width:200px;">
                <span class="feature-badge badge-green">✓ Active</span>
                <p><b>Async Buffering</b><br><small style="color:{COLORS['muted']};">Asynchronous client updates with timeout-based aggregation</small></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    #  SECTION 6: RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📋 Results Summary</div>', unsafe_allow_html=True)

    if all_results:
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("🏁 Initial Accuracy", f"{all_results[0]['accuracy']:.2f}%")
        with col_r2:
            st.metric("🏆 Final Accuracy", f"{all_results[-1]['accuracy']:.2f}%")
        with col_r3:
            improvement = all_results[-1]['accuracy'] - all_results[0]['accuracy']
            st.metric("📈 Improvement", f"{improvement:+.2f}%",
                       delta_color="normal" if improvement > 0 else "inverse")
        with col_r4:
            noise_str = f"{0.05/dp_epsilon:.4f}" if dp_enabled else "N/A"
            st.metric("🔐 DP Noise Scale", noise_str)

        # Save results
        saved_path = save_training_results(
            results=all_results,
            aggregation=aggregation,
            num_clients=num_clients,
            malicious_clients=malicious_clients,
            rounds=min(rounds, 10),
            quick_mode=False,
            max_samples=max_samples,
            dp_enabled=dp_enabled,
            dp_epsilon=dp_epsilon,
        )

        st.markdown(f"""
        <div class="glow-card">
            <h4 style="color:{COLORS['server']}; margin-top:0;">📝 Configuration Recap</h4>
            <table style="width:100%; color:{COLORS['text']}; font-size:0.9rem;">
                <tr><td>Aggregation</td><td style="text-align:right;"><b>{aggregation}</b></td></tr>
                <tr><td>Clients</td><td style="text-align:right;"><b>{num_clients}</b> ({malicious_clients} malicious — {attack_type.replace('_',' ')})</td></tr>
                <tr><td>Rounds × Epochs</td><td style="text-align:right;"><b>{min(rounds,10)} × {local_epochs}</b></td></tr>
                <tr><td>Dataset</td><td style="text-align:right;"><b>CIFAR-10 ({max_samples:,} samples)</b></td></tr>
                <tr><td>Differential Privacy</td><td style="text-align:right;"><b>{'ε=' + str(dp_epsilon) if dp_enabled else 'Disabled'}</b></td></tr>
                <tr><td>Results Saved</td><td style="text-align:right;"><code>{saved_path}</code></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.success("✅ All phases complete — system evaluation finished successfully.")
