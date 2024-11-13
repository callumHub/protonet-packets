import pandas as pd
from torch import nn
import torch
import plotly.express as px
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FeatureVisualizer(object):
    def __init__(self, df: pd.DataFrame, encoder: nn.Module, feature_vectors, feature_names: list, feature_vector_labels):
        self.df = df
        self.feature_vectors = feature_vectors
        self.num_features = len(self.feature_vectors)
        self.n_dim = len(self.feature_vectors[0])
        self.labels = df["labels"]


        self.encoder = encoder
        self.feature_names = feature_names
        self.embeddings = self.__encode_features(self.__proc_features())
        self.feature_vector_labels = feature_vector_labels


    def __proc_features(self):
        x = self.feature_vectors.apply(lambda y: torch.tensor(y, dtype=torch.float32))
        x = torch.cat(x.tolist())
        x = x.view(self.num_features, self.n_dim)
        return x

    def __encode_features(self, x):
        return self.encoder.forward(x)

    def visualize_features(self):
        fig = px.scatter_matrix(
            self.df,
            dimensions=self.feature_names[:10],
            color="labels"
        )
        fig.update_traces(diagonal_visible=False)
        fig.show("browser")

    def visualize_pca_features(self):
        pca = PCA(5)
        components = pca.fit_transform(self.__proc_features())
        labels = {
            str(i): f"PC {i + 1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        fig = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(5),
            color=self.feature_vector_labels,
            title="PCA Raw Features"
        )
        fig.update_traces(diagonal_visible=False)
        fig.show("browser")

    def visualize_pca_features_3d(self):
        pca = PCA(n_components=3)
        components = pca.fit_transform(self.__proc_features())
        total_var = pca.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=self.feature_vector_labels,
            title=f'3d PCA Raw Features (Total Explained Variance: {total_var:.2f}%)',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show("browser")

    def visualize_embeddings(self):
        dimension_names, vis_df = self._prep_embeddings()

        fig = px.scatter_matrix(
            vis_df,
            dimensions=dimension_names[1:],
            color="labels",
            title="Backbone Embeddings"
        )
        fig.update_traces(diagonal_visible=False)
        fig.show("browser")

    def visualize_pca_embeddings(self):
        dimension_names, vis_df = self._prep_embeddings()
        pca = PCA(8)
        components = pca.fit_transform(vis_df[dimension_names[1:]])
        labels = {
            str(i): f"PC {i + 1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        fig = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(8),
            color=vis_df["labels"],
            title="PCA Embeddings (5 components)",
        )
        fig.update_traces(diagonal_visible=False)
        fig.show("browser")

    def visualize_pca_embeddings_3d(self):
        dimension_names, vis_df = self._prep_embeddings()
        pca = PCA(3)
        components = pca.fit_transform(vis_df[dimension_names[1:]])
        total_var = pca.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=vis_df["labels"],
            title=f'3d PCA on Backbone Embeddings (Total Explained Variance: {total_var:.2f}%)',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show("browser")

    def _prep_embeddings(self):
        dimension_names = ["labels"]
        for i in range(self.embeddings.size()[1]):
            dimension_names.append(f"d{i}")
        labels = self.feature_vector_labels
        embs = self.embeddings
        vis_df = pd.DataFrame(labels)
        vis_df[dimension_names[1:]] = embs.detach().numpy()
        return dimension_names, vis_df






