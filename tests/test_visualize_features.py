import unittest
import pandas as pd
from utils.model import load_model
from post_analysis.visualize_features import FeatureVisualizer
class TestVisualizeFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_json(
            "../../enc-vpn-uncertainty-class-repl/processed_data/"+
            "og_test_train_cal/jsonl_versions/data-label-combined.jsonl",
        lines=True)
        self.whole_df = pd.read_json(
            "../../enc-vpn-uncertainty-class-repl/processed_data/"+
            "processed_df_v1-matching.jsonl",
            lines=True)
        self.feature_names = self.whole_df.columns.tolist()[:-2]
        self.model = load_model("../runs/outs/pnet.pt")
        self.model.eval()
        self.encoder = self.model.encoder
        self.fv = FeatureVisualizer(df=self.whole_df, encoder=self.encoder, feature_vectors=self.df.data,
                                    feature_names=self.feature_names, feature_vector_labels=self.df.labels)

    def test_visualize_features(self):
        self.fv.visualize_features()

    def test_visualize_embeddings(self):
        self.fv.visualize_embeddings()

    def test_visualize_pca_embeddings(self):
        self.fv.visualize_pca_embeddings()

    def test_3d_visualize_pca_embeddings(self):
        self.fv.visualize_pca_embeddings_3d()

    def test_3d_visualize_pca_features(self):
        self.fv.visualize_pca_features_3d()

    def test_visualize_pca_features(self):
        self.fv.visualize_pca_features()


    def tearDown(self):
        self.df = None
        self.encoder = None
