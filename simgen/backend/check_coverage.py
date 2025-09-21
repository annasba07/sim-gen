#!/usr/bin/env python
"""
Explain how test coverage is calculated in SimGen AI project.
"""

import json
import sys
from pathlib import Path

def explain_coverage():
    """Explain coverage calculation methodology."""

    print("=" * 60)
    print("TEST COVERAGE CALCULATION METHODOLOGY")
    print("=" * 60)
    print()

    print("Coverage is calculated using Coverage.py (pytest-cov)")
    print()
    print("Formula: Coverage % = (Executed Lines / Total Lines) * 100")
    print()
    print("Where:")
    print("- Total Lines = All executable statements in source code")
    print("- Executed Lines = Lines that ran during test execution")
    print("- Missing Lines = Total Lines - Executed Lines")
    print()

    # Try to load current coverage data
    coverage_file = Path("coverage.json")
    if coverage_file.exists():
        with open(coverage_file) as f:
            data = json.load(f)

        totals = data.get("totals", {})

        total_statements = totals.get("num_statements", 0)
        covered_lines = totals.get("covered_lines", 0)
        missing_lines = totals.get("missing_lines", 0)
        percent = totals.get("percent_covered", 0)

        print("CURRENT COVERAGE METRICS:")
        print("-" * 40)
        print(f"Total Statements: {total_statements:,}")
        print(f"Covered Lines:    {covered_lines:,}")
        print(f"Missing Lines:    {missing_lines:,}")
        print(f"Coverage %:       {percent:.2f}%")
        print()

        # Calculate what's needed for 70%
        target_percent = 70
        needed_lines = int(total_statements * target_percent / 100)
        lines_to_add = needed_lines - covered_lines

        print(f"TARGET: {target_percent}% Coverage")
        print("-" * 40)
        print(f"Lines needed for {target_percent}%: {needed_lines:,}")
        print(f"Additional lines to cover: {lines_to_add:,}")
        print(f"Progress toward target: {(percent/target_percent)*100:.1f}%")
        print()

        # Show top uncovered files
        print("TOP UNCOVERED MODULES:")
        print("-" * 40)

        files = data.get("files", {})
        file_coverage = []

        for filepath, file_data in files.items():
            if "src/simgen" in filepath.replace("\\", "/"):
                summary = file_data.get("summary", {})
                file_coverage.append({
                    "file": filepath.split("src\\simgen\\")[-1] if "src\\simgen\\" in filepath else filepath,
                    "statements": summary.get("num_statements", 0),
                    "missing": summary.get("missing_lines", 0),
                    "percent": summary.get("percent_covered", 0)
                })

        # Sort by number of missing lines (highest first)
        file_coverage.sort(key=lambda x: x["missing"], reverse=True)

        for i, fc in enumerate(file_coverage[:10], 1):
            print(f"{i:2}. {fc['file']:<40} {fc['statements']:>5} lines, {fc['percent']:>5.1f}% covered, {fc['missing']:>5} missing")

    else:
        print("No coverage.json file found. Run tests with --cov flag first.")

    print()
    print("COVERAGE TYPES:")
    print("-" * 40)
    print("• Line Coverage: We're measuring this - which lines executed")
    print("• Branch Coverage: NOT measured - all if/else paths")
    print("• Function Coverage: Partially - if function is called")
    print("• Path Coverage: NOT measured - all execution paths")
    print()

    print("HOW TO IMPROVE COVERAGE:")
    print("-" * 40)
    print("1. Write tests that execute uncovered code paths")
    print("2. Import and instantiate classes (adds ~5-10%)")
    print("3. Call methods with various inputs")
    print("4. Test error handling and edge cases")
    print("5. Mock external dependencies to test integration")
    print()

if __name__ == "__main__":
    explain_coverage()