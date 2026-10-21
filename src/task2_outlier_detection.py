"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    def _numeric_columns(self) -> List[str]:
        """Return the list of numeric columns in the current data."""
        return list(self.data.select_dtypes(include=[np.number]).columns)

    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.

        Parameters:
            threshold (float): Z-score threshold for outlier detection

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric = self.data[self._numeric_columns()]
        std = numeric.std()
        # Columns with zero (or undefined) variance cannot contain outliers
        std = std.replace(0, np.nan)
        z_scores = (numeric - numeric.mean()).abs() / std
        mask = (z_scores > threshold).fillna(False).astype(bool)

        self.outlier_masks['zscore'] = mask
        return mask

    def _iqr_bounds(self, multiplier: float = 1.5) -> Tuple[pd.Series, pd.Series]:
        """Compute lower and upper IQR bounds for each numeric column."""
        numeric = self.data[self._numeric_columns()]
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        return q1 - multiplier * iqr, q3 + multiplier * iqr

    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.

        Parameters:
            multiplier (float): IQR multiplier for boundary calculation

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric = self.data[self._numeric_columns()]
        lower, upper = self._iqr_bounds(multiplier)
        mask = ((numeric < lower) | (numeric > upper)).astype(bool)

        self.outlier_masks['iqr'] = mask
        return mask

    def _prepared_numeric_matrix(self) -> pd.DataFrame:
        """Numeric data with missing values filled by the median (needed by sklearn models)."""
        numeric = self.data[self._numeric_columns()]
        return numeric.fillna(numeric.median())

    def detect_outliers_isolation_forest(self, contamination: float = 0.1,
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.

        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        X = self._prepared_numeric_matrix()
        model = IsolationForest(contamination=contamination, random_state=random_state)
        predictions = model.fit_predict(X)

        # Isolation Forest flags whole rows (multivariate outliers): -1 means outlier
        mask = pd.DataFrame({'is_outlier': predictions == -1}, index=self.data.index)

        self.outlier_masks['isolation_forest'] = mask
        return mask

    def detect_outliers_lof(self, n_neighbors: int = 20,
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.

        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        X = self._prepared_numeric_matrix()
        n_neighbors = max(1, min(n_neighbors, len(X) - 1))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        predictions = model.fit_predict(X)

        mask = pd.DataFrame({'is_outlier': predictions == -1}, index=self.data.index)

        self.outlier_masks['lof'] = mask
        return mask

    def _row_mask(self, method: str) -> pd.Series:
        """Collapse an outlier mask to one boolean per row (True if any column is flagged)."""
        return self.outlier_masks[method].any(axis=1)

    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.

        Returns:
            DataFrame with outlier counts for each method
        """
        summary = {}
        n_rows = len(self.data)

        for method, mask in self.outlier_masks.items():
            stats_for_method = {}
            # Per-feature counts (only available for univariate methods)
            for col in self._numeric_columns():
                stats_for_method[col] = int(mask[col].sum()) if col in mask.columns else np.nan

            rows_flagged = int(self._row_mask(method).sum())
            stats_for_method['rows_with_outliers'] = rows_flagged
            stats_for_method['percentage_of_rows'] = rows_flagged / n_rows * 100 if n_rows else 0.0
            summary[method] = stats_for_method

        return pd.DataFrame(summary)

    def _run_detection(self, method: str) -> None:
        """Run a detection method with default parameters."""
        detectors = {
            'zscore': self.detect_outliers_zscore,
            'iqr': self.detect_outliers_iqr,
            'isolation_forest': self.detect_outliers_isolation_forest,
            'lof': self.detect_outliers_lof
        }
        if method not in detectors:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(detectors)}")
        detectors[method]()

    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Parameters:
            method (str): Detection method to use

        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if method not in self.outlier_masks:
            self._run_detection(method)  # raises ValueError for unknown methods

        cleaned = self.data[~self._row_mask(method)].copy()

        self.treated_datasets[f'removed_{method}'] = cleaned
        return cleaned

    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.

        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds

        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        capped = self.data.copy()
        numeric_cols = self._numeric_columns()

        if method == 'zscore':
            # For z-score, multiplier is interpreted as the number of standard deviations
            mean, std = capped[numeric_cols].mean(), capped[numeric_cols].std()
            lower, upper = mean - multiplier * std, mean + multiplier * std
        else:
            lower, upper = self._iqr_bounds(multiplier)

        capped[numeric_cols] = capped[numeric_cols].clip(lower=lower, upper=upper, axis=1)

        self.treated_datasets['capped'] = capped
        return capped

    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.

        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        transformed = self.data.copy()

        for col in self._numeric_columns():
            values = transformed[col]
            min_value = values.min()
            # Shift so that all values are >= 0, then log1p keeps them finite
            if min_value < 0:
                values = values - min_value
            transformed[col] = np.log1p(values)

        self.treated_datasets['log_transform'] = transformed
        return transformed

    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.

        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        transformed = self.data.copy()
        numeric_cols = self._numeric_columns()

        if numeric_cols:
            # Yeo-Johnson works with zero and negative values, unlike classic Box-Cox
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            filled = transformed[numeric_cols].fillna(transformed[numeric_cols].median())
            transformed[numeric_cols] = transformer.fit_transform(filled)

        self.treated_datasets['boxcox_transform'] = transformed
        return transformed

    def evaluate_model_performance(self, target_col: str,
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.

        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)

        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == 'original':
            df = self.data
        elif dataset_name in self.treated_datasets:
            df = self.treated_datasets[dataset_name]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found. "
                             f"Available: ['original'] + {list(self.treated_datasets)}")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        df = df.select_dtypes(include=[np.number]).dropna()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        cv = min(5, len(df))
        scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='r2')

        return {
            'r2_score': float(scores.mean()),
            'r2_std': float(scores.std()),
            'mean_cv_score': float(scores.mean()),
            'cv_scores': scores.tolist(),
            'n_samples': len(df)
        }

    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.

        Parameters:
            figsize (Tuple[int, int]): Figure size for plots

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric_cols = self._numeric_columns()[:6]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.boxplot(self.data[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
            ax.set_title(f'Box Plot: {col}', fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Outlier Detection: Box Plots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.

        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            values = self.data[col].dropna()
            parts = ax.violinplot(values, showmeans=True, showmedians=True)
            for body in parts['bodies']:
                body.set_facecolor('mediumpurple')
                body.set_alpha(0.6)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('green')
            ax.set_title(f'Violin Plot: {col}', fontweight='bold')
            ax.set_ylabel(col)
            ax.set_xticks([])
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Distribution Density (red = mean, green = median)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def _row_mask_for_columns(self, method: str, columns: List[str]) -> pd.Series:
        """Row-level mask for a method, limited to the given columns for univariate methods."""
        mask = self.outlier_masks[method]
        cols = [c for c in columns if c in mask.columns]
        if cols:
            return mask[cols].any(axis=1)
        return mask.any(axis=1)

    def visualize_outliers_scatter(self, x_col: str, y_col: str,
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.

        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """
        for col in (x_col, y_col):
            if col not in self.data.columns:
                print(f"Column '{col}' not found")
                return

        methods = ['zscore', 'iqr', 'isolation_forest']
        titles = {'zscore': 'Z-Score', 'iqr': 'IQR', 'isolation_forest': 'Isolation Forest'}

        fig, axes = plt.subplots(1, len(methods), figsize=figsize)

        for ax, method in zip(axes, methods):
            if method not in self.outlier_masks:
                ax.text(0.5, 0.5, f'{titles[method]}\nnot computed', ha='center', va='center',
                        fontsize=12, transform=ax.transAxes)
                ax.set_title(titles[method], fontweight='bold')
                continue

            is_outlier = self._row_mask_for_columns(method, [x_col, y_col])
            normal = self.data[~is_outlier]
            outliers = self.data[is_outlier]

            ax.scatter(normal[x_col], normal[y_col], c='steelblue', alpha=0.6, s=20, label='Normal')
            ax.scatter(outliers[x_col], outliers[y_col], c='red', alpha=0.9, s=40,
                       edgecolors='black', label=f'Outlier ({len(outliers)})')
            ax.set_title(titles[method], fontweight='bold')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(f'Outliers: {x_col} vs {y_col}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str,
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.

        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        from mpl_toolkits.mplot3d import Axes3D

        if method not in self.outlier_masks:
            print(f"Outlier mask for method '{method}' not found. Run detection first.")
            return

        for col in (x_col, y_col, z_col):
            if col not in self.data.columns:
                print(f"Column '{col}' not found")
                return

        is_outlier = self._row_mask_for_columns(method, [x_col, y_col, z_col])
        normal = self.data[~is_outlier]
        outliers = self.data[is_outlier]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(normal[x_col], normal[y_col], normal[z_col], c='steelblue', alpha=0.5, s=20,
                   label='Normal')
        ax.scatter(outliers[x_col], outliers[y_col], outliers[z_col], c='red', alpha=0.9, s=50,
                   edgecolors='black', label=f'Outlier ({len(outliers)})')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'3D Outliers ({method.replace("_", " ").title()})', fontweight='bold')
        ax.legend()

        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
