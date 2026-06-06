"""
System health check and dependency verification.
Verifies that all required tools and libraries are installed.
"""
import sys
import subprocess
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)"


def check_python_package(package_name: str) -> Tuple[bool, str]:
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        pkg = sys.modules[package_name]
        version = getattr(pkg, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "not installed"


def check_system_command(command: str) -> Tuple[bool, str]:
    """Check if a system command is available."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version from output (first line usually)
            version = result.stdout.split("\n")[0].strip()
            return True, version
        return False, "command failed"
    except FileNotFoundError:
        return False, "not found"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def run_healthcheck(verbose: bool = True) -> Dict:
    """
    Run complete system health check.

    Args:
        verbose: Print detailed output

    Returns:
        Dict with check results

    Example:
        >>> results = run_healthcheck()
        >>> if not results["all_passed"]:
        ...     print("Some checks failed!")
    """
    results = {
        "python": {},
        "packages": {},
        "system": {},
        "all_passed": True
    }

    if verbose:
        print("=" * 60)
        print("ROOF-METRICS SYSTEM HEALTH CHECK")
        print("=" * 60)

    # Check Python version
    if verbose:
        print("\n[1] Python Version")
        print("-" * 60)

    ok, version = check_python_version()
    results["python"]["version"] = {"ok": ok, "info": version}

    if verbose:
        status = "✓" if ok else "✗"
        print(f"{status} {version}")

    if not ok:
        results["all_passed"] = False

    # Check Python packages
    if verbose:
        print("\n[2] Python Packages")
        print("-" * 60)

    packages = [
        "geopandas",
        "shapely",
        "pyproj",
        "rasterio",
        "pyogrio",
        "pdal",
        "requests",
        "boto3",
        "numpy",
        "scipy",
        "sklearn",
        "cv2",
        "PIL",
        "folium",
        "tqdm",
        "dotenv"
    ]

    for pkg in packages:
        ok, version = check_python_package(pkg)
        results["packages"][pkg] = {"ok": ok, "version": version}

        if verbose:
            status = "✓" if ok else "✗"
            print(f"{status} {pkg:20s} {version}")

        if not ok:
            results["all_passed"] = False

    # Check system tools
    if verbose:
        print("\n[3] System Tools")
        print("-" * 60)

    tools = ["gdal-config", "pdal"]

    for tool in tools:
        ok, version = check_system_command(tool)
        results["system"][tool] = {"ok": ok, "info": version}

        if verbose:
            status = "✓" if ok else "✗"
            print(f"{status} {tool:20s} {version}")

        if not ok and verbose:
            print(f"    Note: {tool} is recommended but not required")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        if results["all_passed"]:
            print("✓ ALL REQUIRED CHECKS PASSED")
        else:
            print("✗ SOME CHECKS FAILED")
            print("\nTo install missing packages:")
            print("  pip install -r requirements.txt")
            print("\nFor system tools (macOS):")
            print("  brew install gdal pdal")
            print("\nFor system tools (Ubuntu/Debian):")
            print("  sudo apt-get install gdal-bin pdal")
        print("=" * 60)

    return results


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run system health check")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()

    results = run_healthcheck(verbose=not args.quiet)

    if args.json:
        import json
        print(json.dumps(results, indent=2))

    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
