import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import torch

from model.protonet_with_target import ProtonetWithTarget
from utils.experiment_context import bandwidth_value
from utils.model import load_model
from model.protonet import get_kde
from utils.parameter_store import HyperParameterStore
from runs.run_train import init_dataloaders, train_pnet
from utils.mahal_utils import Mahalanobis
import utils.experiment_context
from scipy.integrate import quad
from KDEpy import bw_selection
import time
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
utils.experiment_context.bandwidth_experiment = False
utils.experiment_context.bandwidth_value =  0.045
full_path = "../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run0/frac_80"
params = HyperParameterStore().get_model_params("vpn_3hidden", "vpn_models")
# Example: simulate your data
n_class = 5  # Number of classes
n_examples = 100 # Large number of examples per class
data_loader, val_loader = init_dataloaders(full_path, params)
# initialize protonet
pnet = load_model(params.x_dim, params.hidden_dim, params.z_dim, params.dropout, params.hidden_layers,
                  "../runs/outs/vpn_3hidden.pt")
#raw_pnet = ProtonetWithTarget(encoder=pnet.encoder, update_frequency=params.update_frequency, tau=params.tau, params=params)

pnet.load_state_dict(torch.load("../runs/outs/vpn_3hidden.pt"))
pnet.eval()
#pnet.train()
z_s = None
x_q = None
for d in data_loader:
    xs = d["xs"]
    xq = d["xq"]
    break
n_support = len(xs[0])
n_cal = len(xq[0])
x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
               xq.view(n_class * n_cal, *xq.size()[2:])], 0)
with torch.no_grad():
    z = pnet.encoder.forward(x)
z_support = z[:n_class * n_support].view(n_class, n_support, 64)
z_cal = z[n_class * n_support:].view(n_class, n_cal, 64)
rmd = Mahalanobis(z_support, n_cal)  # Now compute mahalanobis # USE support to set mahal vars
# use calibrate to compute rel_mahal (if using sup, ood scores at test time will be higher
#m_k_rel = torch.min(rmd.relative_mahalanobis_distance(z_cal), dim=1).values.view(n_class, n_cal)
m_k_rel = rmd.relative_mahalanobis_distance(z_cal).view(n_class, n_cal, -1).min(dim=0).values.view(n_class, n_cal)
max_val = m_k_rel.max()
min_val = m_k_rel.min()
test_point = min_val-bandwidth_value
print(np.mean([bw_selection.improved_sheather_jones(np.asarray(m_k_rel[i]).reshape(-1,1)) for i in range(n_class)]))
g_k = get_kde(m_k_rel, [], n_class)


# Define a common grid for KDE evaluation:

x_grid = np.linspace(test_point,max_val+(max_val-min_val)*0.541, 75000)

# Create subplots: one per class in a grid layout
n_cols = 2
n_rows = int(np.ceil(n_class / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), sharex=True, sharey=True)
axes = axes.flatten()
data = {f"Class {i}": np.random.normal(loc=0, scale=1.0, size=n_examples) for i in range(n_class)}
ind = 0
for ax, (label, samples) in zip(axes, data.items()):
    # Create the KDE object
    kde = g_k[ind]
    ind += 1
    # Evaluate the density on the grid
    density = kde(x_grid)

    # Plot the KDE curve
    plt.xlim(min_val+(max_val-min_val)*0.1, max_val+(max_val-min_val)*0.541)
    ax.plot(x_grid, density, label=label)
    ax.fill_between(x_grid, density, alpha=0.5)
    ax.set_title(label)
    ax.legend()

# Remove any unused subplots
for i in range(len(data), len(axes)):
    fig.delaxes(axes[i])

# INTEGRATION TIME TESTING
time1 = time.time()
#print(f"quad(g_k[3].pdf, a=min_val, b=max_val, limit=50000), prints: {quad(g_k[3].pdf, a=min_val-0.05, b=max_val+0.05, limit=500000, epsabs=1e-15, epsrel=1e-15)}")
#print("quad time: ", time.time() - time1)
#print(f"quad(g_k[3].pdf, a=-1, b=1, limit=50000), prints: {quad(g_k[3].pdf, a=-1, b=1, limit=500)}")
#print(f"quad(g_k[3].pdf, a=-np.inf, b=np.inf, limit=50000), prints: {quad(g_k[3].pdf, a=-np.inf, b=np.inf, limit=500)}")
time2= time.time()
print([np.trapz(g_k[i].pdf(x_grid), x_grid, dx=0.00001) for i in range(len(g_k))])
print("TRAPZ TIME", time.time()-time2)
time1 = time.time()
print(torch.trapz(y=torch.tensor(g_k[3].pdf(x_grid)), x=torch.tensor(x_grid)).item())
print("TORCH TIME", time.time()-time1)

plt.tight_layout()
plt.show()