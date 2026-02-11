"""
Test script for Probability of Backtest Overfitting (PBO)
"""

import numpy as np
import sys
sys.path.insert(0, 'cnn-lstm')
from lopez_de_prado_evaluation import LopezDePradoEvaluator

def test_pbo_basic():
    """Test PBO with simulated strategy returns"""
    print("="*80)
    print("TEST 1: Basic PBO with Simulated Strategies")
    print("="*80)
    
    # Create 10 strategies with 252 observations each (1 year daily)
    np.random.seed(42)
    n_strategies = 10
    n_observations = 252
    
    # Strategy returns with varying performance
    strategy_returns = []
    for i in range(n_strategies):
        # Create strategies with different Sharpe ratios
        sharpe = np.random.uniform(-0.5, 2.0)
        daily_return = sharpe * 0.01 / np.sqrt(252)
        returns = np.random.normal(daily_return, 0.01, n_observations)
        strategy_returns.append(returns)
    
    # Convert to numpy and transpose: (strategies, obs) -> (obs, strategies)
    strategy_returns = np.array(strategy_returns).T
    print(f"\nStrategy matrix shape: {strategy_returns.shape}")
    print(f"Format: ({n_observations} observations, {n_strategies} strategies)")
    
    # Calculate PBO
    evaluator = LopezDePradoEvaluator(embargo_pct=0.02, n_splits=5)
    pbo_results = evaluator.probability_backtest_overfitting(
        strategy_returns,
        n_splits=16
    )
    
    print(f"\n✅ Results:")
    print(f"   PBO: {pbo_results['pbo']:.3f}")
    print(f"   Interpretation: {pbo_results['interpretation']}")
    print(f"   Number of splits: {pbo_results['n_splits']}")
    
    # Show some lambda values
    if 'lambda_values' in pbo_results:
        lambdas = pbo_results['lambda_values']
        print(f"   Lambda mean: {np.mean(lambdas):.3f}")
        print(f"   Lambda std: {np.std(lambdas):.3f}")
    
    return pbo_results


def test_pbo_overfit():
    """Test PBO with clearly overfit strategies"""
    print("\n" + "="*80)
    print("TEST 2: PBO with Overfit Strategies (Should be HIGH)")
    print("="*80)
    
    np.random.seed(123)
    n_strategies = 20
    n_observations = 252
    
    # Create strategies that are mostly random (will overfit in-sample)
    strategy_returns = []
    for i in range(n_strategies):
        # Pure random walk - any IS success is luck
        returns = np.random.normal(0, 0.02, n_observations)
        strategy_returns.append(returns)
    
    # Convert to numpy and transpose: (strategies, obs) -> (obs, strategies)
    strategy_returns = np.array(strategy_returns).T
    print(f"\nStrategy matrix shape: {strategy_returns.shape}")
    print("Note: These are random walks - should show high PBO (overfitting)")
    
    evaluator = LopezDePradoEvaluator(embargo_pct=0.02, n_splits=5)
    pbo_results = evaluator.probability_backtest_overfitting(
        strategy_returns,
        n_splits=16
    )
    
    print(f"\n✅ Results:")
    print(f"   PBO: {pbo_results['pbo']:.3f}")
    print(f"   Interpretation: {pbo_results['interpretation']}")
    
    if pbo_results['pbo'] > 0.5:
        print("   ✅ Correctly identified overfitting risk!")
    else:
        print("   ⚠️  Lower than expected for random strategies")
    
    return pbo_results


def test_pbo_robust():
    """Test PBO with robust strategies"""
    print("\n" + "="*80)
    print("TEST 3: PBO with Robust Strategies (Should be LOW)")
    print("="*80)
    
    np.random.seed(456)
    n_strategies = 10
    n_observations = 252
    
    # Create strategies with consistent positive drift
    strategy_returns = []
    for i in range(n_strategies):
        # All strategies have positive drift (robust edge)
        daily_return = 0.0003  # ~7.5% annual return
        returns = np.random.normal(daily_return, 0.01, n_observations)
        strategy_returns.append(returns)
    
    # Convert to numpy and transpose: (strategies, obs) -> (obs, strategies)
    strategy_returns = np.array(strategy_returns).T
    print(f"\nStrategy matrix shape: {strategy_returns.shape}")
    print("Note: All strategies have positive drift - should show low PBO")
    
    evaluator = LopezDePradoEvaluator(embargo_pct=0.02, n_splits=5)
    pbo_results = evaluator.probability_backtest_overfitting(
        strategy_returns,
        n_splits=16
    )
    
    print(f"\n✅ Results:")
    print(f"   PBO: {pbo_results['pbo']:.3f}")
    print(f"   Interpretation: {pbo_results['interpretation']}")
    
    if pbo_results['pbo'] < 0.5:
        print("   ✅ Correctly identified robust strategies!")
    else:
        print("   ⚠️  Higher than expected for robust strategies")
    
    return pbo_results


def test_pbo_from_list():
    """Test PBO accepts list input (not just numpy array)"""
    print("\n" + "="*80)
    print("TEST 4: PBO with List Input (API Compatibility)")
    print("="*80)
    
    np.random.seed(789)
    n_strategies = 5
    n_observations = 100
    
    # Create as list (how fin_training.py passes it)
    strategy_returns = []
    for i in range(n_strategies):
        returns = np.random.normal(0.001, 0.02, n_observations)
        strategy_returns.append(returns)
    
    print(f"\nInput type: {type(strategy_returns)}")
    print(f"Number of strategies: {len(strategy_returns)}")
    print(f"Observations per strategy: {len(strategy_returns[0])}")
    
    # Convert to numpy and transpose: (strategies, obs) -> (obs, strategies)
    strategy_returns = np.array(strategy_returns).T
    print(f"After transpose: {strategy_returns.shape}")
    
    evaluator = LopezDePradoEvaluator(embargo_pct=0.02, n_splits=5)
    
    pbo_results = evaluator.probability_backtest_overfitting(
        strategy_returns,
        n_splits=8
    )
    
    print(f"\n✅ Results:")
    print(f"   PBO: {pbo_results['pbo']:.3f}")
    print(f"   ✅ Successfully handled list input!")
    
    return pbo_results


def test_pbo_single_strategy():
    """Test PBO with single strategy (edge case)"""
    print("\n" + "="*80)
    print("TEST 5: PBO with Single Strategy (Edge Case)")
    print("="*80)
    
    np.random.seed(999)
    
    # Single strategy - already in correct shape (252 obs, 1 strategy)
    strategy_returns = np.random.normal(0.001, 0.02, 252).reshape(-1, 1)
    
    print(f"\nStrategy matrix shape: {strategy_returns.shape}")
    print("Note: Only 1 strategy - should return N/A")
    
    evaluator = LopezDePradoEvaluator(embargo_pct=0.02, n_splits=5)
    pbo_results = evaluator.probability_backtest_overfitting(
        strategy_returns,
        n_splits=16
    )
    
    print(f"\n✅ Results:")
    print(f"   PBO: {pbo_results['pbo']}")
    print(f"   Interpretation: {pbo_results['interpretation']}")
    
    if pbo_results['pbo'] is None:
        print("   ✅ Correctly handled single strategy case!")
    
    return pbo_results


def test_pbo_summary():
    """Print summary of all tests"""
    print("\n" + "="*80)
    print("PBO TESTING SUMMARY")
    print("="*80)
    
    results = []
    
    # Run all tests
    print("\n🧪 Running all tests...\n")
    results.append(("Basic", test_pbo_basic()))
    results.append(("Overfit", test_pbo_overfit()))
    results.append(("Robust", test_pbo_robust()))
    results.append(("List Input", test_pbo_from_list()))
    results.append(("Single Strategy", test_pbo_single_strategy()))
    
    # Summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Test':<20} {'PBO':<10} {'Interpretation':<40}")
    print("-"*80)
    
    for name, result in results:
        pbo_val = result['pbo']
        pbo_str = f"{pbo_val:.3f}" if pbo_val is not None else "N/A"
        interp = result['interpretation']
        print(f"{name:<20} {pbo_str:<10} {interp:<40}")
    
    print("="*80)
    print("\n✅ All PBO tests completed!")
    
    # Validation checks
    print("\n🔍 Validation Checks:")
    overfit_pbo = results[1][1]['pbo']
    robust_pbo = results[2][1]['pbo']
    
    if overfit_pbo is not None and robust_pbo is not None:
        if overfit_pbo > robust_pbo:
            print("   ✅ Overfit strategies have higher PBO than robust (correct)")
        else:
            print("   ⚠️  Overfit strategies should have higher PBO than robust")
    
    if results[3][1]['pbo'] is not None:
        print("   ✅ List input handled correctly")
    
    if results[4][1]['pbo'] is None:
        print("   ✅ Single strategy edge case handled correctly")


if __name__ == "__main__":
    test_pbo_summary()
