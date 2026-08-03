"""
Global configuration for IoMT IDS experiments.
All hyperparameters, paths, and constants in one place.
"""
import os
import numpy as np

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("IOMT_DATA_DIR", os.path.join(BASE_DIR, "data"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
XAI_DIR = os.path.join(RESULTS_DIR, "xai")

for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, XAI_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
TEST_SIZE = 0.2      # train/test split from original dataset structure
VAL_SIZE = 0.2       # validation split from training set

# =============================================================================
# ATTACK CATEGORY MAPPINGS (CICIoMT2024)
# =============================================================================
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign',
    # Bluetooth BLE
    'BLE-DoS': 'BLE-DoS',
    'BLE_DoS': 'BLE-DoS',
    'Bluetooth': 'BLE-DoS',
}

ATTACK_CATEGORIES_6 = {
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT', 'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT', 'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon', 'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon', 'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS', 'DDoS-SYN': 'DDoS', 'DDoS-TCP': 'DDoS', 'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS', 'DoS-SYN': 'DoS', 'DoS-TCP': 'DoS', 'DoS-UDP': 'DoS',
    'BLE-DoS': 'DoS',
    'Benign': 'Benign',
}

ATTACK_CATEGORIES_2 = {k: ('Benign' if k == 'Benign' else 'attack')
                        for k in ATTACK_CATEGORIES_19}

CATEGORY_MAP = {2: ATTACK_CATEGORIES_2, 6: ATTACK_CATEGORIES_6, 19: ATTACK_CATEGORIES_19}

# =============================================================================
# FEATURE NAMES (CICIoMT2024)
# =============================================================================
# 45 nomes reais das features numéricas do CICIoMT2024
CICIOMT_FEATURE_NAMES = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Random Forest
RF_PARAMS = dict(
    n_estimators=100, max_depth=None, min_samples_split=2,
    max_features='sqrt', n_jobs=-1, random_state=SEED,
)

# Logistic Regression
LR_PARAMS = dict(
    l1_ratio=0, C=1.0, solver='lbfgs', max_iter=1000,
    random_state=SEED,
)

# LightGBM - Original (defaults, as actually used in the notebook)
LGBM_ORIGINAL_PARAMS = dict(
    n_estimators=100, learning_rate=0.1, num_leaves=31,
    max_depth=-1, boosting_type='gbdt',
    random_state=SEED, n_jobs=-1, verbose=-1,
)

# LightGBM - Improved (addressing Reviewer D criticism)
LGBM_IMPROVED_CONFIGS = {
    'balanced': dict(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, boosting_type='gbdt',
        is_unbalance=True, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    ),
    'aggressive': dict(
        n_estimators=800, learning_rate=0.01, num_leaves=127,
        max_depth=-1, boosting_type='gbdt',
        is_unbalance=True, min_child_samples=5,
        reg_alpha=0.05, reg_lambda=0.05,
        subsample=0.7, colsample_bytree=0.7,
        min_split_gain=0.01,
        random_state=SEED, n_jobs=-1, verbose=-1,
    ),
    'dart': dict(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=15, boosting_type='dart',
        is_unbalance=True, min_child_samples=10, drop_rate=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    ),
}

# 1D-CNN (Standard)
CNN_PARAMS = dict(
    filters_1=32, filters_2=64, kernel_size=3, pool_size=2,
    dense_units=128, dropout=0.0,
    batch_size=512, epochs=10, learning_rate=0.001,
    early_stop_patience=3, mixed_precision=True,
    shuffle_buffer=100_000,
)

# Light-CNN (Depthwise Separable)
LIGHT_CNN_PARAMS = dict(
    filters_1=16, filters_2=32, kernel_size=3, pool_size=2,
    dense_units=64, dropout=0.3,
    batch_size=512, epochs=15, learning_rate=0.001,
    early_stop_patience=3, mixed_precision=True,
    shuffle_buffer=100_000,
)

# CNN com Focal Loss
CNN_FOCAL_PARAMS = dict(
    filters_1=64, filters_2=128, kernel_size=3, pool_size=2,
    dense_units=256, dropout=0.3,
    batch_size=512, epochs=30, learning_rate=0.001,
    early_stop_patience=5, mixed_precision=True,
    shuffle_buffer=100_000,
    focal_gamma=2.0, focal_alpha=0.25,
)

# MLP Baseline (Fully-Connected, sem convoluções)
MLP_BASELINE_PARAMS = dict(
    dense_1=128, dense_2=64, dropout=0.3,
    batch_size=512, epochs=30, learning_rate=0.001,
    early_stop_patience=5, mixed_precision=True,
    shuffle_buffer=100_000,
)

# Light-CNN V2 (mesma arquitetura, mais épocas)
LIGHT_CNN_V2_PARAMS = dict(
    filters_1=16, filters_2=32, kernel_size=3, pool_size=2,
    dense_units=64, dropout=0.3,
    batch_size=512, epochs=30, learning_rate=0.001,
    early_stop_patience=7, mixed_precision=True,
    shuffle_buffer=100_000,
)

# Autoencoder
AUTOENCODER_PARAMS = dict(
    encoding_dims=[64, 32, 16],  # Encoder layers
    dropout=0.2,
    batch_size=512, epochs=20, learning_rate=0.001,
    early_stop_patience=5,
    # Threshold percentile for anomaly detection on validation set
    threshold_percentile=95,
)

# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
MCNEMAR_ALPHA = 0.05
FRIEDMAN_ALPHA = 0.05

# =============================================================================
# EXPLAINABILITY
# =============================================================================
SHAP_BACKGROUND_SAMPLES = 1000
SHAP_TEST_SAMPLES = 500
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 5000

# =============================================================================
# VISUALIZATION
# =============================================================================
FIG_DPI = 600
FIG_FORMAT = ['pdf', 'png']  # Save in both formats
FIG_FONT_SIZE = 11
FIG_TITLE_SIZE = 13

# Publication-quality matplotlib settings
PLOT_STYLE = {
    'font.size': FIG_FONT_SIZE,
    'axes.titlesize': FIG_TITLE_SIZE,
    'axes.labelsize': FIG_FONT_SIZE,
    'xtick.labelsize': FIG_FONT_SIZE - 1,
    'ytick.labelsize': FIG_FONT_SIZE - 1,
    'legend.fontsize': FIG_FONT_SIZE - 1,
    'figure.dpi': FIG_DPI,
    'savefig.dpi': FIG_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palette (colorblind-friendly)
MODEL_COLORS = {
    'Random Forest': '#2196F3',
    'LightGBM (Original)': '#FF9800',
    'LightGBM (Tuned)': '#E65100',
    'Logistic Regression': '#9C27B0',
    '1D-CNN': '#4CAF50',
    'Light-CNN': '#00BCD4',
    'Autoencoder': '#F44336',
    'CNN-FocalLoss': '#795548',
    'Light-CNN V2': '#607D8B',
    'MLP Baseline': '#FF5722',
}
