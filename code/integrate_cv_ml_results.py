#!/usr/bin/env python3
"""
CV-ML Integration Pipeline
=========================

Integrates computer vision indicators with the existing ML model to:
1. Add CV features to the original dataset
2. Retrain Random Forest with enhanced feature set  
3. Analyze how CV indicators explain previous residual anomalies
4. Generate updated anomaly classifications

Expected workflow:
    1. Run download_street_view_images.py
    2. Run process_street_view_segmentation.py  
    3. Run this script to integrate results

Input: 
    - cv_results/data/grid_level_indicators.csv (from CV pipeline)
    - original ML dataset with residuals
Output:
    - enhanced_model_results.csv
    - anomaly_explanation_analysis.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import json
from datetime import datetime
import os

def load_cv_indicators():
    """Load CV indicators from segmentation processing"""
    cv_file = 'cv_results/data/grid_level_indicators.csv'
    
    if not os.path.exists(cv_file):
        print(f"❌ CV indicators file not found: {cv_file}")
        print("   Run process_street_view_segmentation.py first")
        return None
    
    cv_df = pd.read_csv(cv_file)
    print(f"✓ Loaded CV indicators for {len(cv_df)} grids")
    return cv_df

def load_original_ml_data():
    """Load original ML dataset with residuals"""
    possible_files = [
        'ml_with_residuals_filtered.csv',
        '/home/claude/ml_with_residuals_filtered.csv',
        '/home/claude/all_map_data.csv',
        'philadelphia_heat_data.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✓ Loaded original ML data: {len(df)} grids from {file_path}")
            return df
    
    print("❌ Original ML dataset not found")
    return None

def merge_cv_with_ml_data(ml_df, cv_df):
    """Merge CV indicators with original ML dataset"""
    
    # Merge on GRID_ID
    merged = ml_df.merge(cv_df, left_on='GRID_ID', right_on='grid_id', how='left')
    
    print(f"📊 Merge results:")
    print(f"   Original grids: {len(ml_df)}")
    print(f"   Grids with CV data: {len(cv_df)}")
    print(f"   Successfully merged: {merged.dropna(subset=['gvi']).shape[0]}")
    
    # Fill missing CV indicators with dataset means (for grids without street view)
    cv_columns = ['gvi', 'svf', 'bvf', 'rvf', 'canyon_ratio', 'canopy_height_proxy', 'vegetation_sky_ratio']
    for col in cv_columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].mean())
    
    return merged, cv_columns

def retrain_enhanced_model(df, original_features, cv_features, target='Mean_LST'):
    """Retrain Random Forest with CV-enhanced feature set"""
    
    # Prepare feature sets
    baseline_features = original_features
    enhanced_features = baseline_features + cv_features
    
    # Remove any missing values
    df_clean = df.dropna(subset=enhanced_features + [target])
    
    print(f"\n🤖 Training enhanced Random Forest model")
    print(f"   Baseline features: {len(baseline_features)}")
    print(f"   CV features: {len(cv_features)}")  
    print(f"   Total features: {len(enhanced_features)}")
    print(f"   Training samples: {len(df_clean)}")
    
    # Train baseline model
    X_baseline = df_clean[baseline_features]
    y = df_clean[target]
    
    baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
    baseline_scores = cross_val_score(baseline_model, X_baseline, y, cv=5, scoring='r2')
    baseline_model.fit(X_baseline, y)
    
    # Train enhanced model  
    X_enhanced = df_clean[enhanced_features]
    
    enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
    enhanced_scores = cross_val_score(enhanced_model, X_enhanced, y, cv=5, scoring='r2')
    enhanced_model.fit(X_enhanced, y)
    
    # Calculate predictions and residuals
    baseline_pred = baseline_model.predict(X_baseline)
    enhanced_pred = enhanced_model.predict(X_enhanced)
    
    baseline_residuals = y - baseline_pred
    enhanced_residuals = y - enhanced_pred
    
    # Feature importance analysis
    baseline_importance = dict(zip(baseline_features, baseline_model.feature_importances_))
    enhanced_importance = dict(zip(enhanced_features, enhanced_model.feature_importances_))
    
    results = {
        'baseline_r2': np.mean(baseline_scores),
        'enhanced_r2': np.mean(enhanced_scores),
        'r2_improvement': np.mean(enhanced_scores) - np.mean(baseline_scores),
        'baseline_rmse': np.sqrt(mean_squared_error(y, baseline_pred)),
        'enhanced_rmse': np.sqrt(mean_squared_error(y, enhanced_pred)),
        'baseline_importance': baseline_importance,
        'enhanced_importance': enhanced_importance,
        'models': {
            'baseline': baseline_model,
            'enhanced': enhanced_model
        }
    }
    
    # Add predictions and residuals to dataframe
    df_clean = df_clean.copy()
    df_clean['baseline_pred'] = baseline_pred
    df_clean['enhanced_pred'] = enhanced_pred
    df_clean['baseline_residual'] = baseline_residuals
    df_clean['enhanced_residual'] = enhanced_residuals
    df_clean['residual_improvement'] = np.abs(baseline_residuals) - np.abs(enhanced_residuals)
    
    return results, df_clean

def analyze_anomaly_explanations(df, cv_features, residual_threshold=1.5):
    """Analyze how CV indicators explain original anomalies"""
    
    # Identify original anomalies
    residual_std = df['baseline_residual'].std()
    original_anomalies = df[np.abs(df['baseline_residual']) > residual_threshold * residual_std].copy()
    
    print(f"\n🔍 Analyzing {len(original_anomalies)} original anomalies")
    
    # Calculate correlation between CV indicators and residual reduction
    cv_correlations = {}
    for feature in cv_features:
        if feature in df.columns:
            corr = df['residual_improvement'].corr(df[feature])
            cv_correlations[feature] = corr
    
    # Find best explanations for each anomaly type
    hot_anomalies = original_anomalies[original_anomalies['baseline_residual'] > 0]
    cool_anomalies = original_anomalies[original_anomalies['baseline_residual'] < 0]
    
    explanations = {
        'hot_anomalies': {
            'count': len(hot_anomalies),
            'mean_cv_indicators': {feat: hot_anomalies[feat].mean() 
                                 for feat in cv_features if feat in hot_anomalies.columns},
            'top_explanatory_features': sorted(cv_correlations.items(), 
                                             key=lambda x: -abs(x[1]))[:3]
        },
        'cool_anomalies': {
            'count': len(cool_anomalies), 
            'mean_cv_indicators': {feat: cool_anomalies[feat].mean()
                                 for feat in cv_features if feat in cool_anomalies.columns},
            'top_explanatory_features': sorted(cv_correlations.items(),
                                             key=lambda x: -abs(x[1]))[:3]
        },
        'overall_correlations': cv_correlations
    }
    
    return explanations

def save_results(model_results, enhanced_df, explanations, cv_features):
    """Save all integration results"""
    
    # Model comparison results
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'model_comparison': {
            'baseline_r2': model_results['baseline_r2'],
            'enhanced_r2': model_results['enhanced_r2'], 
            'r2_improvement': model_results['r2_improvement'],
            'baseline_rmse': model_results['baseline_rmse'],
            'enhanced_rmse': model_results['enhanced_rmse']
        },
        'cv_features_added': cv_features,
        'feature_importance': {
            'baseline_top3': sorted(model_results['baseline_importance'].items(), 
                                  key=lambda x: -x[1])[:3],
            'enhanced_top3': sorted(model_results['enhanced_importance'].items(),
                                  key=lambda x: -x[1])[:3],
            'cv_feature_importance': {feat: model_results['enhanced_importance'][feat] 
                                    for feat in cv_features 
                                    if feat in model_results['enhanced_importance']}
        },
        'anomaly_explanations': explanations
    }
    
    # Save detailed results
    os.makedirs('integration_results', exist_ok=True)
    
    with open('integration_results/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    enhanced_df.to_csv('integration_results/enhanced_dataset.csv', index=False)
    
    # Create summary report
    summary = f"""
CV-ML Integration Results
========================
Timestamp: {comparison['timestamp']}

Model Performance:
  Baseline R²: {model_results['baseline_r2']:.4f}
  Enhanced R²: {model_results['enhanced_r2']:.4f}
  Improvement: +{model_results['r2_improvement']:.4f}
  
  Baseline RMSE: {model_results['baseline_rmse']:.3f}°C
  Enhanced RMSE: {model_results['enhanced_rmse']:.3f}°C

Top CV Feature Importance:
"""
    
    cv_importance = {feat: model_results['enhanced_importance'][feat] 
                    for feat in cv_features 
                    if feat in model_results['enhanced_importance']}
    
    for feat, importance in sorted(cv_importance.items(), key=lambda x: -x[1]):
        summary += f"  {feat}: {importance:.4f}\n"
    
    summary += f"\nAnomaly Explanation Power:\n"
    for feat, corr in sorted(explanations['overall_correlations'].items(), key=lambda x: -abs(x[1])):
        summary += f"  {feat}: {corr:.3f} correlation with residual reduction\n"
    
    with open('integration_results/summary_report.txt', 'w') as f:
        f.write(summary)
    
    return comparison

def main():
    """Main integration pipeline"""
    print("🔗 CV-ML Integration Pipeline")
    print("============================")
    
    # Load data
    cv_df = load_cv_indicators()
    ml_df = load_original_ml_data()
    
    if cv_df is None or ml_df is None:
        return
    
    # Merge datasets
    merged_df, cv_features = merge_cv_with_ml_data(ml_df, cv_df)
    
    if len(cv_features) == 0:
        print("❌ No CV features found to integrate")
        return
    
    # Define original ML features (adapt based on your dataset)
    original_features = ['MEAN_Impervious', 'MEAN_Canopy', 'PCT_Building Coverage',
                        'sum_Road_Length', 'Pct_MinorE', 'Pct_belowp',
                        'Density_To', 'Pct_18', 'Pct_65']
    
    # Filter to existing columns
    original_features = [f for f in original_features if f in merged_df.columns]
    cv_features = [f for f in cv_features if f in merged_df.columns]
    
    if not original_features:
        print("⚠ Warning: No original features found, using available numeric columns")
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        original_features = [col for col in numeric_cols 
                           if col not in cv_features + ['LST', 'residual', 'GRID_ID']][:5]
    
    print(f"\n📊 Feature sets:")
    print(f"   Original features ({len(original_features)}): {original_features}")
    print(f"   CV features ({len(cv_features)}): {cv_features}")
    
    # Retrain models
    model_results, enhanced_df = retrain_enhanced_model(
        merged_df, original_features, cv_features
    )
    
    # Analyze explanations
    explanations = analyze_anomaly_explanations(enhanced_df, cv_features)
    
    # Save results
    comparison = save_results(model_results, enhanced_df, explanations, cv_features)
    
    # Print summary
    print(f"\n📈 MODEL IMPROVEMENT SUMMARY")
    print(f"===========================")
    print(f"R² improvement: {model_results['r2_improvement']:+.4f}")
    print(f"RMSE improvement: {model_results['baseline_rmse'] - model_results['enhanced_rmse']:+.3f}°C")
    print(f"\nTop CV feature importance:")
    cv_importance = {feat: model_results['enhanced_importance'][feat] 
                    for feat in cv_features 
                    if feat in model_results['enhanced_importance']}
    for feat, imp in sorted(cv_importance.items(), key=lambda x: -x[1])[:3]:
        print(f"  {feat}: {imp:.4f}")
    
    print(f"\n📁 Results saved to:")
    print(f"   integration_results/enhanced_dataset.csv")
    print(f"   integration_results/model_comparison.json") 
    print(f"   integration_results/summary_report.txt")
    
    print(f"\n✅ Integration complete!")

if __name__ == "__main__":
    main()
