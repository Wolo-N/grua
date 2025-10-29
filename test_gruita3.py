#!/usr/bin/env python3
"""
TEST SCRIPT FOR GRUITA 3
========================
This script loads the best crane design saved by gruita3.py and runs tests on it.

Usage:
------
1. First run gruita3.py to optimize and save the best crane design
2. Then run this script to load and test the saved design

Example:
--------
python test_gruita3.py

Or to specify a specific crane file:
python test_gruita3.py path/to/best_crane.pkl
"""

import sys
from pathlib import Path
from gruita3 import load_crane, analyze_moving_load, animate_moving_load
import numpy as np

def main():
    print("="*80)
    print("GRUITA 3 - TESTING SAVED CRANE DESIGN")
    print("="*80)

    # Get crane file path from command line or use default
    if len(sys.argv) > 1:
        crane_file = Path(sys.argv[1])
    else:
        # Find the most recent gruita3_iterations directory
        iteration_dirs = sorted(Path('.').glob('gruita3_iterations_*'))
        if not iteration_dirs:
            print("\nERROR: No gruita3_iterations_* directories found!")
            print("Please run gruita3.py first to generate a crane design.")
            return

        # Use the most recent directory
        latest_dir = iteration_dirs[-1]
        crane_file = latest_dir / 'best_crane.pkl'

        if not crane_file.exists():
            print(f"\nERROR: No best_crane.pkl found in {latest_dir}")
            print("Please run gruita3.py first to generate a crane design.")
            return

    print(f"\nLoading crane from: {crane_file}")
    print("-"*80)

    # Load the crane design
    crane_data = load_crane(crane_file)

    # Print design summary
    print("\n" + "="*80)
    print("DESIGN SUMMARY")
    print("="*80)
    diag_names = {
        0: "Alternating (Warren)",
        1: "All positive slope",
        2: "All negative slope",
        3: "X-pattern",
        4: "Fan from bottom",
        5: "Fan from top",
        6: "Long-span",
        7: "Mixed span",
        8: "Concentrated",
        9: "Progressive fan",
        10: "Full connectivity"
    }

    print(f"Segments: {crane_data['n_segments']}")
    print(f"Boom height: {crane_data['boom_height']:.3f} m")
    print(f"Taper ratio: {crane_data['taper_ratio']:.3f}")
    diag_num = int(crane_data['connectivity_pattern'][0])
    print(f"Diagonal type: {diag_num} - {diag_names.get(diag_num, 'Unknown')}")
    print(f"Vertical spacing: every {int(crane_data['connectivity_pattern'][1])} nodes")
    print(f"Support cables: {int(crane_data['connectivity_pattern'][2])}")
    print(f"\nTotal elements: {crane_data['C'].shape[0]}")
    print(f"Total nodes: {crane_data['X'].shape[0]}")
    print(f"Objective value: {crane_data['objective']:.2f}")
    print("="*80)

    # Menu for user
    while True:
        print("\n" + "="*80)
        print("TESTING OPTIONS")
        print("="*80)
        print("1. Run moving load analysis (0-40 kN)")
        print("2. Animate moving load (30 kN)")
        print("3. Custom load test")
        print("4. Exit")
        print("="*80)

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            print("\nRunning moving load analysis with loads from 0 to 40 kN...")
            analyze_moving_load(crane_data)

        elif choice == '2':
            print("\nCreating animation of 30 kN load moving along the crane...")
            animate_moving_load(crane_data, load_magnitude=30000, scale_factor=100, interval=200)

        elif choice == '3':
            try:
                load = float(input("Enter load magnitude in kN: "))
                load_N = load * 1000

                print("\nWhat would you like to do?")
                print("1. Analyze multiple load positions")
                print("2. Animate the moving load")

                sub_choice = input("Enter your choice (1-2): ").strip()

                if sub_choice == '1':
                    analyze_moving_load(crane_data, load_magnitudes=np.array([load_N]))
                elif sub_choice == '2':
                    animate_moving_load(crane_data, load_magnitude=load_N, scale_factor=100, interval=200)
                else:
                    print("Invalid choice!")

            except ValueError:
                print("Invalid input! Please enter a number.")

        elif choice == '4':
            print("\nExiting test script. Goodbye!")
            break

        else:
            print("Invalid choice! Please enter a number between 1 and 4.")

if __name__ == '__main__':
    main()
