import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model_202512 import GR2ST
from dataset import HERDataset, TenxDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R
import sys
from types import SimpleNamespace
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def generate_args():
    args = SimpleNamespace()
    args.batch_size = 256
    args.max_epochs = 201
    args.temperature = 1.0
    args.fold = 0
    args.dim = 50
    args.image_embedding_dim = 1024
    args.projection_dim = 256
    args.heads_num = 8
    args.heads_dim = 64
    args.heads_layers = 2
    args.dropout = 0.1
    args.dataset = '10x'
    args.encoder_name = 'densenet121'
    args.alpha_mse = 50.0
    args.alpha_gate = 1.0
    args.alpha_entropy = 0.01
    args.spatial_radius = 3.0
    args.conf_threshold = 0.6
    args.fusion_type = 'sum'
    return args

def build_loaders_inference():
    loaders = []
    examples = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]

    for example in examples:
        dataset = TenxDataset(
            image_path=f"./data/Alex_NatGen/{example}/{example}.tif",
            spatial_pos_path=f"./data/Alex_NatGen/{example}/spatial/tissue_positions_list.csv",
            reduced_mtx_path=f"./data/Alex_NatGen/preprocessed_expression_matrices/Alex/{example}/preprocessed_matrix.npy",
            barcode_path=f"./data/Alex_NatGen/{example}/filtered_count_matrix/barcodes.tsv.gz",
            precomputed_features_dir=f"./data/Alex_NatGen/preprocessed_data/precomputed_features/{example}",
            cell_type_path=f"./data/Alex_NatGen/spots_type/{example}/summary_spot_types.tsv",
            train=False
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        loaders.append(loader)
    print("Finished building Tenx loaders")
    return loaders, examples

def get_embeddings(model_path, model, test_loaders):
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_key = new_key.replace('well', 'spot')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to('cuda')
    print(f"Finished loading model from {model_path}")

    test_image_embeddings = []
    spot_embeddings = []
    regression_predictions = []

    with torch.no_grad():
        for loader in test_loaders:
            for batch in loader:
                batch = {k: v.cuda() for k, v in batch.items() if
                         k in ["image_features", "expression", "position", "cell_type"]}

                image_features = batch["image_features"]
                image_embeddings = model.image_projection(image_features)
                test_image_embeddings.append(image_embeddings)

                reg_pred = model.predict_expression(image_features)
                regression_predictions.append(reg_pred)

                spot_feature = batch["expression"]
                x = batch["position"][:, 0].long()
                y = batch["position"][:, 1].long()
                centers_x = model.x_embed(x)
                centers_y = model.y_embed(y)
                cell_type = batch["cell_type"].long()
                cell_type_embeddings = model.cell_type_embed(cell_type)

                spot_features = spot_feature + centers_x + centers_y + cell_type_embeddings

                positions_float = batch["position"].float()
                dynamic_edge_index = model.build_threshold_functional_graph(spot_features, cell_type)
                spatial_edge_index = model.build_radius_spatial_graph(positions_float)

                dynamic_features = model.dynamic_head_proj(spot_features)
                spatial_features = model.spatial_proj(spot_features)

                out_dyn = model.dynamic_gat(dynamic_features, dynamic_edge_index)
                if isinstance(out_dyn, tuple):
                    out_dyn = out_dyn[0]

                out_spa = model.spatial_gat(spatial_features, spatial_edge_index)
                if isinstance(out_spa, tuple):
                    out_spa = out_spa[0]

                if model.fusion_type == 'concat':
                    fused_features = torch.cat([out_dyn, out_spa], dim=-1)
                elif model.fusion_type == 'sum':
                    fused_features = out_dyn + out_spa
                elif model.fusion_type == 'max':
                    fused_features = torch.max(out_dyn, out_spa)
                else:
                    fused_features = (out_dyn + out_spa) / 2

                fused_features = model.fusion_layer(fused_features)
                spot_embedding = model.spot_projection(fused_features)
                spot_embeddings.append(spot_embedding)

    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings), torch.cat(regression_predictions)

def find_matches(spot_embeddings, query_embeddings, top_k=1):
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    return indices.cpu().numpy()

args = generate_args()

names = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]

print("Loading ground truth data...")
spot_expressions = [np.load(f"./data/Alex_NatGen/preprocessed_expression_matrices/Alex/{name}/preprocessed_matrix.npy")
                    for name in names]
datasize = [mat.shape[1] for mat in spot_expressions]

test_loaders, examples = build_loaders_inference()

all_hvg_pcc_list = []
all_heg_pcc_list = []
all_mse_list = []
all_mae_list = []

for fold in range(6):
    print(f"=== Evaluating Fold {fold}: {names[fold]} ===")

    model = GR2ST(
        temperature=args.temperature,
        image_dim=args.image_embedding_dim,
        spot_dim=args.dim,
        projection_dim=args.projection_dim,
        heads_num=args.heads_num,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        alpha_mse=args.alpha_mse,
        alpha_gate=args.alpha_gate,
        alpha_entropy=args.alpha_entropy,
        spatial_radius=args.spatial_radius,
        conf_threshold=args.conf_threshold
    )

    model_path = f"./model_result_202512/10x/{names[fold]}/best_{fold}.pt"

    img_embeddings_all, spot_embeddings_all, regression_all = get_embeddings(model_path, model, test_loaders)

    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    regression_all = regression_all.cpu().numpy()

    spot_embeddings = []
    regression_pred_fold = None
    image_embeddings = None

    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])

        spot_embeddings0 = spot_embeddings_all[index_start:index_end]
        spot_embeddings.append(spot_embeddings0.T)

        if i == fold:
            image_embeddings = img_embeddings_all[index_start:index_end].T
            regression_pred_fold = regression_all[index_start:index_end]

    image_query = image_embeddings
    expression_gt = spot_expressions[fold]

    spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold + 1:]
    spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold + 1:]

    spot_key = np.concatenate(spot_embeddings, axis=1)
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    if image_query.shape[1] != 256:
        image_query = image_query.T
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T

    indices = find_matches(spot_key, image_query, top_k=200)
    matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))

    for i in range(indices.shape[0]):
        a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
        reciprocal_of_square_a = np.reciprocal(a ** 2 + 1e-8)
        weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
        matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights.flatten())

    if regression_pred_fold.shape[0] != matched_spot_expression_pred.shape[0]:
        regression_pred_fold = regression_pred_fold.T

    final_pred = 0.4 * matched_spot_expression_pred + 0.6 * regression_pred_fold
    dir = f"./alex_pred_att/{names[fold]}/"
    os.makedirs(dir, exist_ok=True)
    np.save(f"./alex_pred_att/{names[fold]}/" + "matched_spot_expression_pred_202512.npy", final_pred.T)
    true = expression_gt
    pred = final_pred

    slice_name = names[fold]
    slice_dir = os.path.join("./st_results_202512/alex/")
    os.makedirs(slice_dir, exist_ok=True)

    gene_list_path = "./data/Alex_NatGen/Alex.npy"
    if os.path.exists(gene_list_path):
        gene_list = list(np.load(gene_list_path, allow_pickle=True))
        if len(gene_list) == pred.shape[1]:
            pred_df = pd.DataFrame(pred, columns=gene_list)
            pred_df.to_csv(os.path.join(slice_dir, f"{slice_name}_pred.csv"), index=False)
            true_df = pd.DataFrame(true, columns=gene_list)
    else:
        gene_list = [f"Gene_{i}" for i in range(true.shape[1])]

    gene_pcc_list = []
    for j in range(true.shape[1]):
        true_j = true[:, j]
        pred_j = pred[:, j]
        pcc, _ = pearsonr(true_j, pred_j)
        gene_name = gene_list[j] if j < len(gene_list) else f"Gene_{j}"
        gene_pcc_list.append((gene_name, pcc))

    gene_pcc_list.sort(key=lambda x: x[1], reverse=True)

    pcc_path = os.path.join(slice_dir, f"{slice_name}_corr.csv")
    with open(pcc_path, 'w') as f:
        f.write("Gene,PCC\n")
        for gene, pcc in gene_pcc_list:
            f.write(f"{gene},{pcc}\n")

    mse_list = []
    for j in range(true.shape[1]):
        mse = mean_squared_error(true[:, j], pred[:, j])
        gene_name = gene_list[j] if j < len(gene_list) else f"Gene_{j}"
        mse_list.append((gene_name, mse))
    mse_list.sort(key=lambda x: x[1])

    with open(os.path.join(slice_dir, f"{slice_name}_mse.csv"), 'w') as f:
        f.write("Gene,MSE\n")
        for gene, mse in mse_list:
            f.write(f"{gene},{mse}\n")

    mae_list = []
    for j in range(true.shape[1]):
        mae = mean_absolute_error(true[:, j], pred[:, j])
        gene_name = gene_list[j] if j < len(gene_list) else f"Gene_{j}"
        mae_list.append((gene_name, mae))
    mae_list.sort(key=lambda x: x[1])

    with open(os.path.join(slice_dir, f"{slice_name}_mae.csv"), 'w') as f:
        f.write("Gene,MAE\n")
        for gene, mae in mae_list:
            f.write(f"{gene},{mae}\n")

    print(f"Saved metrics for {slice_name}. Avg PCC: {np.mean([x[1] for x in gene_pcc_list]):.4f}")

    adata_ture = anndata.AnnData(true)
    adata_pred = anndata.AnnData(pred)
    if len(gene_list) == true.shape[1]:
        adata_pred.var_names = gene_list
        adata_ture.var_names = gene_list

    gene_mean_expression = np.mean(adata_ture.X, axis=0)
    top_50_indices = np.argsort(gene_mean_expression)[::-1][:50]
    heg_pcc_val, _ = get_R(adata_pred[:, top_50_indices], adata_ture[:, top_50_indices])
    heg_pcc_val = heg_pcc_val[~np.isnan(heg_pcc_val)]
    all_heg_pcc_list.append(np.mean(heg_pcc_val))

    hvg_pcc_val, _ = get_R(adata_pred, adata_ture)
    hvg_pcc_val = hvg_pcc_val[~np.isnan(hvg_pcc_val)]
    all_hvg_pcc_list.append(np.mean(hvg_pcc_val))

    all_mse_list.append(mean_squared_error(true, pred))
    all_mae_list.append(mean_absolute_error(true, pred))

print("=== Final Results ===")
print(f"Avg HEG PCC: {np.mean(all_heg_pcc_list):.4f}")
print(f"Avg HVG PCC: {np.mean(all_hvg_pcc_list):.4f}")
print(f"Mean Squared Error (MSE): {np.mean(all_mse_list):.4f}")
print(f"Mean Absolute Error (MAE): {np.mean(all_mae_list):.4f}")

for i in range(6):
    print(i)
    print(f"Avg HEG PCC: {all_heg_pcc_list[i]:.4f}")
    print(f"Avg HVG PCC: {all_hvg_pcc_list[i]:.4f}")
    print(f"Mean Squared Error (MSE): {all_mse_list[i]:.4f}")
    print(f"Mean Absolute Error (MAE): {all_mae_list[i]:.4f}")