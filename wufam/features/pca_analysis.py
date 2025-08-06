from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Tuple, Optional


class PCAAnalysis:    
    def __init__(self, n_components: Optional[int] = None) -> None:
        """
        Initialize PCA Analysis.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If None, keep all components.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
        # Fitted attributes
        self._components = None
        self._explained_variance_ratio = None
        self._explained_variance = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._mean = None
        self._feature_names = None
        self._covariance_matrix = None
        self._data = None
        self._pc_scores = None
        self._risk_prices = None
        
    def fit(self, returns_data: pd.DataFrame) -> 'PCAAnalysis':
        self._feature_names = returns_data.columns.tolist()
        self._data = returns_data.copy()
        
        # Calculate mean for centering (PCA still requires centering)
        self._mean = returns_data.mean()
        
        # Covariance matrix calculation (data will be centered by sklearn PCA)
        self._covariance_matrix = returns_data.cov()
        
        # Eigendecomposition using V = EΛE^T
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix.values)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors[:, idx]
        print(f"Eigenvalues: {self._eigenvalues.round(4)}")
        
        self.pca.fit(returns_data)
        
        # Store fitted attributes
        self._components = self.pca.components_
        self._explained_variance_ratio = self.pca.explained_variance_ratio_
        self._explained_variance = self.pca.explained_variance_

        print("Explained Variance Ratio:")
        print(self._explained_variance_ratio)
        
        # Calculate PC scores and risk prices during fitting
        pc_scores = self.pca.transform(returns_data)
        self._pc_scores = pd.DataFrame(
            pc_scores,
            index=returns_data.index,
            columns=[f'PC{i+1}' for i in range(pc_scores.shape[1])]
        )
        
        self.is_fitted = True
        return self
        
    # Get principal components and explained variance
    def get_components(self, n_components: int = 3) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting components")
            
        components_df = pd.DataFrame(
            self._components[:n_components],
            columns=self._feature_names,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        return components_df
        
    def get_pc_scores(self, n_components: int = 3) -> pd.DataFrame:
        """Get principal component scores (factor realizations)"""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting PC scores")
            
        # Return the pre-calculated PC scores
        return self._pc_scores.iloc[:, :n_components]
        
    def estimate_risk_prices(self, n_components: int = 3) -> pd.Series:
        """Get the pre-calculated risk prices (γ = sample average of PCs)"""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting risk prices")
            
        # Return the pre-calculated risk prices
        return self._risk_prices.iloc[:n_components]
        
    def plot_component_loadings(self, n_components: int = 3) -> plt.Figure:
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before plotting")
            
        components_df = self.get_components(n_components)
        
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4 * n_components))
        if n_components == 1:
            axes = [axes]
            
        for i, (pc_name, loadings) in enumerate(components_df.iterrows()):
            ax = axes[i]
            bars = ax.bar(range(len(loadings)), loadings.values)
            ax.set_title(f'{pc_name} Loadings (Explained Variance: {self._explained_variance_ratio[i]:.3f})')
            ax.set_xlabel('Assets')
            ax.set_ylabel('Loading')
            ax.set_xticks(range(len(loadings)))
            ax.set_xticklabels(loadings.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Color bars by sign
            for bar, value in zip(bars, loadings.values):
                if value >= 0:
                    bar.set_color('steelblue')
                else:
                    bar.set_color('orange')
        
        plt.tight_layout()
        return fig
        
    def analyze_factor_relationship(self, 
                                   factors_df: pd.DataFrame,
                                   n_components: int = 3) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before analyzing factor relationships")
            
        # Get PC scores using the pre-calculated scores
        pc_df = self.get_pc_scores(n_components)
        
        # Align indices (use intersection of dates)
        common_dates = pc_df.index.intersection(factors_df.index)
        pc_aligned = pc_df.loc[common_dates]
        factors_aligned = factors_df.loc[common_dates]
        
        # Calculate correlations
        correlations = pd.DataFrame(index=pc_aligned.columns, columns=factors_aligned.columns)
        
        for pc in pc_aligned.columns:
            for factor in factors_aligned.columns:
                correlations.loc[pc, factor] = pc_aligned[pc].corr(factors_aligned[factor])
                
        return correlations.astype(float)
        
    def plot_factor_correlations(self, 
                                correlations: pd.DataFrame,
                                title: str = "PC-Factor Correlations") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(correlations, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Factors')
        ax.set_ylabel('Principal Components')
        
        plt.tight_layout()
        return fig