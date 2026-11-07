import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import copy




class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, output_dim, scale, mapping):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        self.mapping = mapping
        B_matrix = None

        if self.mapping == "gaussian":
            B_matrix = torch.normal(mean=0, std=scale, size=(output_dim, input_dim))
            self.output_dim = output_dim * 2
        elif self.mapping == "basic":
            B_matrix = torch.eye(input_dim)
            self.output_dim = input_dim * 2
        else:
            raise ValueError(f"Mapping \"{self.mapping}\" is not implemented.\nPlease check the documentation for the implemented mappings.")

        self.register_buffer('B', B_matrix)

    def forward(self, x):
        theta = 2 * torch.pi * (x @ self.B.t())     # order reversed since input will be batched. (N, d) @ (m, d)^T = (N, m)
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)


class FourierFeatureNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, scale=10, mapping="gaussian", mapping_dim=256, layers=4, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.hidden_dim = hidden_dim

        self.ffm = FourierFeatureMapping(input_dim, mapping_dim, scale, mapping)
        mlp_layers = []
        layer_in_dim = self.ffm.output_dim
        for _ in range(self.layers):
            mlp_layers.append(nn.Linear(layer_in_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            layer_in_dim = hidden_dim

        mlp_layers.append(nn.Linear(layer_in_dim, output_dim))
        mlp_layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.ffm(x)
        return self.mlp(x)


def train_ffn(epochs,
              img_pth,
              device,
              img_size=(0, 0),
              batch_size=65536,
              input_dim=2,
              output_dim=3,
              scale=10,
              mapping="gaussian",
              mapping_dim=256,
              hidden_dim=256,
              layers=4,
              only_batches_to_vram=True):
    img = Image.open(img_pth).convert('RGB')
    if img_size != (0,0):
        img = img.resize(img_size)
    img_tensor = transforms.ToTensor()(img)
    _, H, W = img_tensor.shape

    x_coords = torch.linspace(0.0, 1.0, W)
    y_coords = torch.linspace(0.0, 1.0, H)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    coord_matrix = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    rgb_matrix = img_tensor.permute(1, 2, 0).reshape(-1, 3)
    if not only_batches_to_vram:
        coord_matrix = coord_matrix.to(device)
        rgb_matrix = rgb_matrix.to(device)

    model = FourierFeatureNetwork(input_dim, output_dim, scale, mapping=mapping, mapping_dim=mapping_dim, layers=layers, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)

    dataset_size = H * W
    for epoch in range(epochs):
        epoch_loss = 0.0
        permutation = torch.randperm(dataset_size, device=device)
        if only_batches_to_vram:
            permutation = permutation.cpu()
        for i in range(0, dataset_size, batch_size):
            indeces = permutation[i:i + batch_size]
            coords, rgb = coord_matrix[indeces], rgb_matrix[indeces]
            if only_batches_to_vram:
                coords = coords.to(device)
                rgb = rgb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(coords), rgb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch: {epoch+1}/{epochs} | Loss: {epoch_loss / ((dataset_size + batch_size - 1) // batch_size):.6f}")

    params = {'h': H,
              'w': W,
              'input_dim': input_dim,
              'output_dim': output_dim,
              'scale': scale,
              'mapping': mapping,
              'mapping_dim': mapping_dim,
              'hidden_dim': hidden_dim,
              'layers': layers}
    return model, params


def save_model(model, params):
    now = datetime.now()
    t_now = now.strftime("%Y_%m_%d_%H_%M")

    os.makedirs("models", exist_ok=True)
    savefile = {'state_dict': model.state_dict(),
                'param_dict': params}
    torch.save(savefile, f'models/FFN2dImReg_{t_now}.pth')


def eval_model(model, device, h_target, w_target, batch_size=8192, only_batches_to_vram=False, dementia_factor=0.0):
    model = copy.deepcopy(model).to(device)
    model.eval()

    x_coords = torch.linspace(0.0, 1.0, w_target)
    y_coords = torch.linspace(0.0, 1.0, h_target)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_matrix = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    if not only_batches_to_vram:
        coords_matrix = coords_matrix.to(device)

    predictions = []
    with torch.no_grad():
        if dementia_factor > 0.0:
            layer_id = 0
            for l in model.mlp:
                if layer_id % 2 == 0:
                    mask = torch.rand_like(model.mlp[layer_id].weight) < dementia_factor
                    model.mlp[layer_id].weight[mask] = 0

                layer_id += 1

        dataset_size = w_target * h_target
        for i in range(0, dataset_size, batch_size):
            coords = coords_matrix[i : i + batch_size]
            if only_batches_to_vram:
                coords = coords.to(device)
            preds = model(coords)
            predictions.append(preds)

    recreated_tensor = torch.cat(predictions, dim=0).view(h_target, w_target, 3).detach().cpu()
    return recreated_tensor


def save_eval_img(img_tensor):
    chw_tensor = img_tensor.permute(2, 0, 1)
    now = datetime.now()
    t_now = now.strftime("%Y_%m_%d_%H_%M")

    os.makedirs("saved_images", exist_ok=True)
    save_image(chw_tensor, f'saved_images/{t_now + "_" + str(np.random.randint(0, 19))}.png')


def plot_eval_img(img_tensor):
    img_np = img_tensor.numpy()
    h, w, _ = img_np.shape

    fig_width = 8
    fig_height = fig_width * (h/w)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img_np, aspect='auto')
    plt.show()


if __name__ == "__main__":      # keep all function calls in the main process, if at least one has more than 0 workers
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_pth = "samples/stillworld.png"


        model, param_dict = train_ffn(100, img_pth, device, scale=15, mapping="gaussian", mapping_dim=256, hidden_dim=256, only_batches_to_vram=False)
        save_model(model, param_dict)
        # savefile = torch.load("models/.pth", weights_only=True)
        # param_dict = savefile['param_dict']
        # model = FourierFeatureNetwork(input_dim=param_dict['input_dim'], output_dim=param_dict['output_dim'], scale=param_dict['scale'], mapping=param_dict['mapping'], mapping_dim=param_dict['mapping_dim'], hidden_dim=param_dict['hidden_dim']).to(device)
        # model.load_state_dict(savefile['state_dict'])
        recreated_img = eval_model(model, device, param_dict['h'], param_dict['w'], only_batches_to_vram=False)
        recreated_img2 = eval_model(model, device, param_dict['h'], param_dict['w'], dementia_factor=0.15, only_batches_to_vram=False)
        save_eval_img(recreated_img)
        save_eval_img(recreated_img2)
        plot_eval_img(recreated_img)
        plot_eval_img(recreated_img2)

    except ValueError as e:
        print(e)