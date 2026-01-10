```python
# Example usage and comprehensive testing
if __name__ == "__main__":
    # Initialize production-ready checker
    checker = ProductionPackageChecker(
        cache_size=1024,
        max_recursion_depth=4,
        enable_parallel=True,
        default_discovery_mode=DiscoveryMode.STANDARD
    )
    
    # Comprehensive test suite
    test_packages = [
        'os', 'sys', 'time',              # Standard library
        'numpy', 'pandas', 'requests',    # Common third-party
        'non_existent_package_xyz'        # Non-existent
    ]
    
    print("=== Production Package Checker Demo ===\n")
    
    # Test 1: Warm-up demonstration
    print("1. Cache Warm-up:")
    warm_up_results = checker.warm_up(['os', 'sys', 'json'], DiscoveryMode.FAST)
    for pkg, success in warm_up_results.items():
        print(f"   {pkg:15} | Warm-up: {'✓' if success else '✗'}")
    
    # Test 2: Individual checks with different modes
    print("\n2. Individual Package Analysis:")
    for pkg in test_packages:
        discovery_mode = DiscoveryMode.PARALLEL if pkg in checker.HEAVY_PACKAGES else DiscoveryMode.FAST
        info = checker.check_package_availability(pkg, discovery_mode=discovery_mode)
        
        print(f"{pkg:20} | Exists: {info.exists:5} | Type: {info.package_type.value:10} | "
              f"Version: {info.version:12} | Submodules: {info.sub_module_count:4} | "
              f"Time: {info.load_time:.3f}s | Mode: {info.discovery_mode.value}")
    
    # Test 3: Batch processing
    print("\n3. Batch Check Performance:")
    batch_start = time.perf_counter()
    batch_results = checker.batch_check_packages(
        test_packages[:5], 
        max_workers=3,
        discovery_mode=DiscoveryMode.FAST
    )
    batch_time = time.perf_counter() - batch_start
    print(f"   Batch processed {len(batch_results)} packages in {batch_time:.3f}s")
    
    # Test 4: Package summaries
    print("\n4. Package Summaries:")
    for pkg in ['os', 'sys']:
        summary = checker.summary(pkg)
        print(f"   {pkg}: {summary}")
    
    # Test 5: Cache performance analysis
    print("\n5. Cache Performance:")
    cache_stats = checker.cache_stats()
    print(f"   Hit Rate: {cache_stats.hit_rate:.1f}%")
    print(f"   Cache Utilization: {cache_stats.used_slots}/{cache_stats.size} "
          f"({cache_stats.used_slots/cache_stats.size*100:.1f}%)")
    print(f"   Memory Usage: {cache_stats.total_memory_mb:.1f} MB")
    
    # Test 6: Overall statistics
    print("\n6. Performance Statistics:")
    stats = checker.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key:25}: {value:.3f}")
        else:
            print(f"   {key:25}: {value}")
    
    # Test 7: Simple API compatibility
    print("\n7. Simple API Compatibility:")
    print(f"   OS module available: {is_package_available('os')}")
    print(f"   OS with version: {is_package_available('os', return_version=True)}")
    print(f"   Non-existent: {is_package_available('non_existent_xyz')}")
    
    # Test 8: Summary API
    print("\n8. Summary API:")
    summary = package_summary('os')
    print(f"   OS Summary: {summary}")
    
    print(f"\n=== Production demo completed successfully ===")

```