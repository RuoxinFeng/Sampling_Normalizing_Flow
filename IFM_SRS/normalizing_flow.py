import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import normflows as nf
from scipy.stats import gaussian_kde


def normflow(training_data, hidden_units, hidden_layers, num_flows, max_iter, batch_size, lr):
    seed = 42 
    cols = training_data.columns.tolist()
    cols.remove('type')
    cols.append('type')  
    training_data = training_data[cols] 
    x_data = training_data.iloc[:, :-1].to_numpy().astype(np.float32)
    training_data.iloc[:, -1] = training_data.iloc[:, -1].astype(str)
    context_df = pd.get_dummies(training_data.iloc[:, -1], prefix='type')
    context_data = context_df.to_numpy().astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    context_data = torch.tensor(context_data, dtype=torch.float32).to(device)

    latent_size = x_data.shape[1] 
    context_size = context_data.shape[1] 

    flows = []
    for _ in range(num_flows):
        flows.append(
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units, 
                num_context_channels=context_size
            )
        )
        flows.append(nf.flows.LULinearPermute(latent_size))

    base = nf.distributions.DiagGaussian(latent_size, trainable=False)
    target = nf.distributions.target.ConditionalDiagGaussian()
    model = nf.ConditionalNormalizingFlow(base, flows, target).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist = []
    best_loss = float('inf')  
    best_model_state = None  

    for i in tqdm(range(max_iter)):
        idx = torch.randint(0, x_data.shape[0], (batch_size,))
        x_batch = x_data[idx]
        context_batch = context_data[idx]

        loss = model.forward_kld(x_batch, context_batch)
        if not torch.isnan(loss) and not torch.isinf(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_hist.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()  
            best_model_state = model.state_dict()  

    model.load_state_dict(best_model_state)

    # Evaluate PDF on training data
    with torch.no_grad():
        log_pdf = model.log_prob(x_data, context_data)
        pdf = torch.exp(log_pdf).cpu().numpy()

    return model, context_df.columns.tolist(), pdf


def plot_pdf_surface(samples_df):
    columns = samples_df.columns.tolist()
    features = columns[:-1]
    pdf_column = columns[-1]

    fig = plt.figure(figsize=(5 * len(features), 5 * len(features)))

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            ax = fig.add_subplot(len(features), len(features), j * len(features) + i + 1, projection='3d')

            x = samples_df[features[i]].values
            y = samples_df[features[j]].values
            z = samples_df[pdf_column].values

            kde = gaussian_kde(np.vstack([x, y, z]))
            x_grid = np.linspace(x.min(), x.max(), 30)
            y_grid = np.linspace(y.min(), y.max(), 30)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = kde(np.vstack([X.ravel(), Y.ravel(), np.mean(z) * np.ones_like(X.ravel())])).reshape(X.shape)

            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            ax.set_title(f'PDF Plot of {features[i]}, {features[j]}')
            ax.set_xlabel(features[i])
            ax.set_ylabel(features[j])
            ax.set_zlabel('PDF')

    plt.tight_layout()
    plt.show()