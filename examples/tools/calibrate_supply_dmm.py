from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from openlifu_sdk.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/calibrate_supply_dmm.py

# ============================================================================
# CALIBRATION COEFFICIENT GENERATION FUNCTIONS
# ============================================================================

def fit_polynomial(x_data, y_data, order=2):
    """Fit polynomial to data and return coefficients and quality metrics"""
    coeffs = np.polyfit(x_data, y_data, order)
    poly_func = np.poly1d(coeffs)
    y_pred = poly_func(x_data)
    errors = y_pred - y_data
    r2 = r2_score(y_data, y_pred)
    max_err = np.max(np.abs(errors))
    rms_err = np.sqrt(np.mean(errors**2))
    return coeffs, r2, max_err, rms_err


def generate_calibration_coefficients(pos_csv_path=None, neg_csv_path=None):
    """Generate calibration coefficients from CSV files

    Returns:
        tuple: (pos_results, neg_results) dictionaries with coefficients
    """
    pos_results = None
    neg_results = None

    # Process positive supply
    if pos_csv_path and Path(pos_csv_path).exists():
        try:
            pos_df = pd.read_csv(pos_csv_path)
            pos_results = {
                'supply_type': 'positive',
                'ch0_name': 'Ch0',
                'ch1_name': 'Ch1',
                'voltage_range': (df['DMM_Voltage_V'].min(), df['DMM_Voltage_V'].max()),
                'dac_range': (int(df['DAC_Value'].min()), int(df['DAC_Value'].max()))
            }

            # Fit polynomials
            coeffs, r2, max_err, rms_err = fit_polynomial(df['Raw_ADC_Ch0'], df['DMM_Voltage_V'])
            pos_results['adc_ch0'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (int(df['Raw_ADC_Ch0'].min()), int(df['Raw_ADC_Ch0'].max()))
            }

            coeffs, r2, max_err, rms_err = fit_polynomial(df['Raw_ADC_Ch1'], df['DMM_Voltage_V'])
            pos_results['adc_ch1'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (int(df['Raw_ADC_Ch1'].min()), int(df['Raw_ADC_Ch1'].max()))
            }

            coeffs, r2, max_err, rms_err = fit_polynomial(df['DAC_Value'], df['DMM_Voltage_V'])
            pos_results['dac_to_volt'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (int(df['DAC_Value'].min()), int(df['DAC_Value'].max()))
            }

            coeffs, r2, max_err, rms_err = fit_polynomial(df['DMM_Voltage_V'], df['DAC_Value'])
            pos_results['volt_to_dac'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (df['DMM_Voltage_V'].min(), df['DMM_Voltage_V'].max())
            }

            print(f"✅ Processed positive supply calibration from {pos_csv_path}")
        except Exception as e:
            print(f"⚠️  Error processing positive calibration: {e}")

    # Process negative supply
    if neg_csv_path and Path(neg_csv_path).exists():
        try:
            neg_df = pd.read_csv(neg_csv_path)
            neg_results = {
                'supply_type': 'negative',
                'ch0_name': 'Ch2',
                'ch1_name': 'Ch3',
                'voltage_range': (df['DMM_Voltage_V'].min(), df['DMM_Voltage_V'].max()),
                'dac_range': (int(df['DAC_Value'].min()), int(df['DAC_Value'].max()))
            }

            # Fit polynomials
            coeffs, r2, max_err, rms_err = fit_polynomial(df['Raw_ADC_Ch2'], df['DMM_Voltage_V'])
            neg_results['adc_ch0'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (int(df['Raw_ADC_Ch2'].min()), int(df['Raw_ADC_Ch2'].max()))
            }

            coeffs, r2, max_err, rms_err = fit_polynomial(df['Raw_ADC_Ch3'], df['DMM_Voltage_V'])
            neg_results['adc_ch1'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (int(df['Raw_ADC_Ch3'].min()), int(df['Raw_ADC_Ch3'].max()))
            }

            coeffs, r2, max_err, rms_err = fit_polynomial(df['DAC_Value'], df['DMM_Voltage_V'])
            neg_results['dac_to_volt'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (int(df['DAC_Value'].min()), int(df['DAC_Value'].max()))
            }

            coeffs, r2, max_err, rms_err = fit_polynomial(df['DMM_Voltage_V'], df['DAC_Value'])
            neg_results['volt_to_dac'] = {
                'coeffs': coeffs, 'r2': r2, 'max_error': max_err, 'rms_error': rms_err,
                'range': (df['DMM_Voltage_V'].min(), df['DMM_Voltage_V'].max())
            }

            print(f"✅ Processed negative supply calibration from {neg_csv_path}")
        except Exception as e:
            print(f"⚠️  Error processing negative calibration: {e}")

    return pos_results, neg_results


def save_python_coefficients(pos_results, neg_results, output_file='calibration_coeffs.py'):
    """Generate Python module with calibration coefficients"""
    lines = []
    lines.append('"""')
    lines.append("Auto-generated High Voltage Calibration Coefficients")
    lines.append("Generated by calibrate_supply_dmm.py")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")

    if pos_results:
        lines.append("# Positive Supply Coefficients")
        lines.append(f"POS_ADC_CH0_COEFFS = np.array([{pos_results['adc_ch0']['coeffs'][0]:.12e}, {pos_results['adc_ch0']['coeffs'][1]:.12e}, {pos_results['adc_ch0']['coeffs'][2]:.12e}])")
        lines.append(f"POS_ADC_CH1_COEFFS = np.array([{pos_results['adc_ch1']['coeffs'][0]:.12e}, {pos_results['adc_ch1']['coeffs'][1]:.12e}, {pos_results['adc_ch1']['coeffs'][2]:.12e}])")
        lines.append(f"POS_DAC_TO_VOLT_COEFFS = np.array([{pos_results['dac_to_volt']['coeffs'][0]:.12e}, {pos_results['dac_to_volt']['coeffs'][1]:.12e}, {pos_results['dac_to_volt']['coeffs'][2]:.12e}])")
        lines.append(f"POS_VOLT_TO_DAC_COEFFS = np.array([{pos_results['volt_to_dac']['coeffs'][0]:.12e}, {pos_results['volt_to_dac']['coeffs'][1]:.12e}, {pos_results['volt_to_dac']['coeffs'][2]:.12e}])")
        lines.append(f"POS_VOLT_RANGE = ({pos_results['voltage_range'][0]:.2f}, {pos_results['voltage_range'][1]:.2f})")
        lines.append(f"POS_DAC_RANGE = ({pos_results['dac_range'][0]}, {pos_results['dac_range'][1]})")
        lines.append("")

    if neg_results:
        lines.append("# Negative Supply Coefficients")
        lines.append(f"NEG_ADC_CH2_COEFFS = np.array([{neg_results['adc_ch0']['coeffs'][0]:.12e}, {neg_results['adc_ch0']['coeffs'][1]:.12e}, {neg_results['adc_ch0']['coeffs'][2]:.12e}])")
        lines.append(f"NEG_ADC_CH3_COEFFS = np.array([{neg_results['adc_ch1']['coeffs'][0]:.12e}, {neg_results['adc_ch1']['coeffs'][1]:.12e}, {neg_results['adc_ch1']['coeffs'][2]:.12e}])")
        lines.append(f"NEG_DAC_TO_VOLT_COEFFS = np.array([{neg_results['dac_to_volt']['coeffs'][0]:.12e}, {neg_results['dac_to_volt']['coeffs'][1]:.12e}, {neg_results['dac_to_volt']['coeffs'][2]:.12e}])")
        lines.append(f"NEG_VOLT_TO_DAC_COEFFS = np.array([{neg_results['volt_to_dac']['coeffs'][0]:.12e}, {neg_results['volt_to_dac']['coeffs'][1]:.12e}, {neg_results['volt_to_dac']['coeffs'][2]:.12e}])")
        lines.append(f"NEG_VOLT_RANGE = ({neg_results['voltage_range'][0]:.2f}, {neg_results['voltage_range'][1]:.2f})")
        lines.append(f"NEG_DAC_RANGE = ({neg_results['dac_range'][0]}, {neg_results['dac_range'][1]})")
        lines.append("")

    lines.append("def pos_adc_ch0_to_voltage(adc_raw):")
    lines.append("    return np.polyval(POS_ADC_CH0_COEFFS, adc_raw)")
    lines.append("")
    lines.append("def pos_adc_ch1_to_voltage(adc_raw):")
    lines.append("    return np.polyval(POS_ADC_CH1_COEFFS, adc_raw)")
    lines.append("")
    lines.append("def pos_voltage_to_dac(voltage):")
    lines.append("    dac = np.polyval(POS_VOLT_TO_DAC_COEFFS, voltage)")
    lines.append("    return int(np.clip(np.round(dac), POS_DAC_RANGE[0], POS_DAC_RANGE[1]))")
    lines.append("")
    lines.append("def neg_adc_ch2_to_voltage(adc_raw):")
    lines.append("    return np.polyval(NEG_ADC_CH2_COEFFS, adc_raw)")
    lines.append("")
    lines.append("def neg_adc_ch3_to_voltage(adc_raw):")
    lines.append("    return np.polyval(NEG_ADC_CH3_COEFFS, adc_raw)")
    lines.append("")
    lines.append("def neg_voltage_to_dac(voltage):")
    lines.append("    dac = np.polyval(NEG_VOLT_TO_DAC_COEFFS, voltage)")
    lines.append("    return int(np.clip(np.round(dac), NEG_DAC_RANGE[0], NEG_DAC_RANGE[1]))")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ Generated Python coefficients: {output_file}")


def save_c_header(pos_results, neg_results, output_file='hv_calibration_coeffs.h'):
    """Generate C header file with calibration coefficients"""
    lines = []
    lines.append("/* Auto-generated High Voltage Calibration Coefficients */")
    lines.append("#ifndef HV_CALIBRATION_COEFFS_H")
    lines.append("#define HV_CALIBRATION_COEFFS_H")
    lines.append("")

    if pos_results:
        lines.append("// Positive Supply - ADC Ch0 -> Voltage")
        lines.append(f"#define POS_CH0_A  ({pos_results['adc_ch0']['coeffs'][0]:.12e}f)")
        lines.append(f"#define POS_CH0_B  ({pos_results['adc_ch0']['coeffs'][1]:.12e}f)")
        lines.append(f"#define POS_CH0_C  ({pos_results['adc_ch0']['coeffs'][2]:.12e}f)")
        lines.append("")
        lines.append("// Positive Supply - ADC Ch1 -> Voltage")
        lines.append(f"#define POS_CH1_A  ({pos_results['adc_ch1']['coeffs'][0]:.12e}f)")
        lines.append(f"#define POS_CH1_B  ({pos_results['adc_ch1']['coeffs'][1]:.12e}f)")
        lines.append(f"#define POS_CH1_C  ({pos_results['adc_ch1']['coeffs'][2]:.12e}f)")
        lines.append("")
        lines.append("// Positive Supply - Voltage -> DAC")
        lines.append(f"#define POS_VOLT_A ({pos_results['volt_to_dac']['coeffs'][0]:.12e}f)")
        lines.append(f"#define POS_VOLT_B ({pos_results['volt_to_dac']['coeffs'][1]:.12e}f)")
        lines.append(f"#define POS_VOLT_C ({pos_results['volt_to_dac']['coeffs'][2]:.12e}f)")
        lines.append("")

    if neg_results:
        lines.append("// Negative Supply - ADC Ch2 -> Voltage")
        lines.append(f"#define NEG_CH2_A  ({neg_results['adc_ch0']['coeffs'][0]:.12e}f)")
        lines.append(f"#define NEG_CH2_B  ({neg_results['adc_ch0']['coeffs'][1]:.12e}f)")
        lines.append(f"#define NEG_CH2_C  ({neg_results['adc_ch0']['coeffs'][2]:.12e}f)")
        lines.append("")
        lines.append("// Negative Supply - ADC Ch3 -> Voltage")
        lines.append(f"#define NEG_CH3_A  ({neg_results['adc_ch1']['coeffs'][0]:.12e}f)")
        lines.append(f"#define NEG_CH3_B  ({neg_results['adc_ch1']['coeffs'][1]:.12e}f)")
        lines.append(f"#define NEG_CH3_C  ({neg_results['adc_ch1']['coeffs'][2]:.12e}f)")
        lines.append("")
        lines.append("// Negative Supply - Voltage -> DAC")
        lines.append(f"#define NEG_VOLT_A ({neg_results['volt_to_dac']['coeffs'][0]:.12e}f)")
        lines.append(f"#define NEG_VOLT_B ({neg_results['volt_to_dac']['coeffs'][1]:.12e}f)")
        lines.append(f"#define NEG_VOLT_C ({neg_results['volt_to_dac']['coeffs'][2]:.12e}f)")
        lines.append("")

    lines.append("#endif")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ Generated C header: {output_file}")


def save_c_implementation(pos_results, neg_results, output_file='hv_calibration_coeffs.c'):
    """Generate C implementation file"""
    lines = []
    lines.append("/* Auto-generated High Voltage Calibration Functions */")
    lines.append('#include "hv_calibration_coeffs.h"')
    lines.append("#include <stdint.h>")
    lines.append("")

    if pos_results:
        lines.append("float pos_adc_ch0_to_voltage(uint16_t adc_raw) {")
        lines.append("    float x = (float)adc_raw;")
        lines.append("    return POS_CH0_A * x * x + POS_CH0_B * x + POS_CH0_C;")
        lines.append("}")
        lines.append("")
        lines.append("float pos_adc_ch1_to_voltage(uint16_t adc_raw) {")
        lines.append("    float x = (float)adc_raw;")
        lines.append("    return POS_CH1_A * x * x + POS_CH1_B * x + POS_CH1_C;")
        lines.append("}")
        lines.append("")
        lines.append("uint16_t pos_voltage_to_dac(float voltage) {")
        lines.append("    float dac_float = POS_VOLT_A * voltage * voltage + POS_VOLT_B * voltage + POS_VOLT_C;")
        lines.append("    int32_t dac = (int32_t)(dac_float + 0.5f);")
        lines.append(f"    if (dac < {pos_results['dac_range'][0]}) return {pos_results['dac_range'][0]};")
        lines.append(f"    if (dac > {pos_results['dac_range'][1]}) return {pos_results['dac_range'][1]};")
        lines.append("    return (uint16_t)dac;")
        lines.append("}")
        lines.append("")

    if neg_results:
        lines.append("float neg_adc_ch2_to_voltage(uint16_t adc_raw) {")
        lines.append("    float x = (float)adc_raw;")
        lines.append("    return NEG_CH2_A * x * x + NEG_CH2_B * x + NEG_CH2_C;")
        lines.append("}")
        lines.append("")
        lines.append("float neg_adc_ch3_to_voltage(uint16_t adc_raw) {")
        lines.append("    float x = (float)adc_raw;")
        lines.append("    return NEG_CH3_A * x * x + NEG_CH3_B * x + NEG_CH3_C;")
        lines.append("}")
        lines.append("")
        lines.append("uint16_t neg_voltage_to_dac(float voltage) {")
        lines.append("    float dac_float = NEG_VOLT_A * voltage * voltage + NEG_VOLT_B * voltage + NEG_VOLT_C;")
        lines.append("    int32_t dac = (int32_t)(dac_float + 0.5f);")
        lines.append(f"    if (dac < {neg_results['dac_range'][0]}) return {neg_results['dac_range'][0]};")
        lines.append(f"    if (dac > {neg_results['dac_range'][1]}) return {neg_results['dac_range'][1]};")
        lines.append("    return (uint16_t)dac;")
        lines.append("}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ Generated C implementation: {output_file}")




def plot_calibration_data(pos_csv_path=None, neg_csv_path=None, output_file='calibration_plots.png'):
    """Generate comprehensive calibration plots"""
    try:
        import matplotlib as mpl
        mpl.use('Agg')  # Use non-interactive backend
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not installed. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(16, 12))
    plot_num = 1

    pos_df = None
    neg_df = None

    # Load data
    if pos_csv_path and Path(pos_csv_path).exists():
        pos_pos_df = pd.read_csv(pos_csv_path)
    if neg_csv_path and Path(neg_csv_path).exists():
        neg_neg_df = pd.read_csv(neg_csv_path)

    if pos_df is None and neg_df is None:
        print("❌ No calibration data found to plot")
        return

    # ========================================================================
    # POSITIVE SUPPLY PLOTS
    # ========================================================================
    if pos_df is not None:
        # Plot 1: DAC vs Voltage
        ax1 = plt.subplot(3, 3, 1)
        ax1.scatter(pos_df['DAC_Value'], pos_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Measured')
        coeffs = np.polyfit(pos_df['DAC_Value'], pos_df['DMM_Voltage_V'], 2)
        dac_fit = np.linspace(pos_df['DAC_Value'].min(), pos_df['DAC_Value'].max(), 200)
        volt_fit = np.polyval(coeffs, dac_fit)
        ax1.plot(dac_fit, volt_fit, 'r-', linewidth=2, label='Polynomial Fit')
        ax1.set_xlabel('DAC Value', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax1.set_title('Positive Supply: DAC → Voltage', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: ADC Ch0 vs Voltage
        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(pos_df['Raw_ADC_Ch0'], pos_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Measured', color='blue')
        coeffs = np.polyfit(pos_df['Raw_ADC_Ch0'], pos_df['DMM_Voltage_V'], 2)
        adc_fit = np.linspace(pos_df['Raw_ADC_Ch0'].min(), pos_df['Raw_ADC_Ch0'].max(), 200)
        volt_fit = np.polyval(coeffs, adc_fit)
        ax2.plot(adc_fit, volt_fit, 'r-', linewidth=2, label='Polynomial Fit')
        ax2.set_xlabel('Raw ADC Ch0', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax2.set_title('Positive Supply: ADC Ch0 → Voltage', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: ADC Ch1 vs Voltage
        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(pos_df['Raw_ADC_Ch1'], pos_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Measured', color='green')
        coeffs = np.polyfit(pos_df['Raw_ADC_Ch1'], pos_df['DMM_Voltage_V'], 2)
        adc_fit = np.linspace(pos_df['Raw_ADC_Ch1'].min(), pos_df['Raw_ADC_Ch1'].max(), 200)
        volt_fit = np.polyval(coeffs, adc_fit)
        ax3.plot(adc_fit, volt_fit, 'r-', linewidth=2, label='Polynomial Fit')
        ax3.set_xlabel('Raw ADC Ch1', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax3.set_title('Positive Supply: ADC Ch1 → Voltage', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 7: Error analysis
        ax7 = plt.subplot(3, 3, 7)
        coeffs = np.polyfit(pos_df['DAC_Value'], pos_df['DMM_Voltage_V'], 2)
        volt_pred = np.polyval(coeffs, pos_df['DAC_Value'])
        errors = volt_pred - pos_df['DMM_Voltage_V']
        ax7.scatter(pos_df['DMM_Voltage_V'], errors, alpha=0.6, s=50, color='red')
        ax7.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax7.set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Error (V)', fontsize=11, fontweight='bold')
        ax7.set_title(f'Positive DAC Error\nMax: {np.max(np.abs(errors)):.3f}V, RMS: {np.sqrt(np.mean(errors**2)):.3f}V',
                      fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)

    # ========================================================================
    # NEGATIVE SUPPLY PLOTS
    # ========================================================================
    if neg_df is not None:
        # Plot 4: DAC vs Voltage
        ax4 = plt.subplot(3, 3, 4)
        ax4.scatter(neg_df['DAC_Value'], neg_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Measured', color='purple')
        coeffs = np.polyfit(neg_df['DAC_Value'], neg_df['DMM_Voltage_V'], 2)
        dac_fit = np.linspace(neg_df['DAC_Value'].min(), neg_df['DAC_Value'].max(), 200)
        volt_fit = np.polyval(coeffs, dac_fit)
        ax4.plot(dac_fit, volt_fit, 'r-', linewidth=2, label='Polynomial Fit')
        ax4.set_xlabel('DAC Value', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax4.set_title('Negative Supply: DAC → Voltage', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Plot 5: ADC Ch2 vs Voltage
        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(neg_df['Raw_ADC_Ch2'], neg_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Measured', color='orange')
        coeffs = np.polyfit(neg_df['Raw_ADC_Ch2'], neg_df['DMM_Voltage_V'], 2)
        adc_fit = np.linspace(neg_df['Raw_ADC_Ch2'].min(), neg_df['Raw_ADC_Ch2'].max(), 200)
        volt_fit = np.polyval(coeffs, adc_fit)
        ax5.plot(adc_fit, volt_fit, 'r-', linewidth=2, label='Polynomial Fit')
        ax5.set_xlabel('Raw ADC Ch2', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax5.set_title('Negative Supply: ADC Ch2 → Voltage', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # Plot 6: ADC Ch3 vs Voltage
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(neg_df['Raw_ADC_Ch3'], neg_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Measured', color='brown')
        coeffs = np.polyfit(neg_df['Raw_ADC_Ch3'], neg_df['DMM_Voltage_V'], 2)
        adc_fit = np.linspace(neg_df['Raw_ADC_Ch3'].min(), neg_df['Raw_ADC_Ch3'].max(), 200)
        volt_fit = np.polyval(coeffs, adc_fit)
        ax6.plot(adc_fit, volt_fit, 'r-', linewidth=2, label='Polynomial Fit')
        ax6.set_xlabel('Raw ADC Ch3', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax6.set_title('Negative Supply: ADC Ch3 → Voltage', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

        # Plot 8: Error analysis
        ax8 = plt.subplot(3, 3, 8)
        coeffs = np.polyfit(neg_df['DAC_Value'], neg_df['DMM_Voltage_V'], 2)
        volt_pred = np.polyval(coeffs, neg_df['DAC_Value'])
        errors = volt_pred - neg_df['DMM_Voltage_V']
        ax8.scatter(neg_df['DMM_Voltage_V'], errors, alpha=0.6, s=50, color='purple')
        ax8.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax8.set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Error (V)', fontsize=11, fontweight='bold')
        ax8.set_title(f'Negative DAC Error\nMax: {np.max(np.abs(errors)):.3f}V, RMS: {np.sqrt(np.mean(errors**2)):.3f}V',
                      fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3)

    # ========================================================================
    # COMBINED PLOT
    # ========================================================================
    ax9 = plt.subplot(3, 3, 9)
    if pos_df is not None:
        ax9.scatter(pos_df['DAC_Value'], pos_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Positive Supply', color='blue')
    if neg_df is not None:
        ax9.scatter(neg_df['DAC_Value'], neg_df['DMM_Voltage_V'], alpha=0.6, s=50, label='Negative Supply', color='red')
    ax9.set_xlabel('DAC Value', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    ax9.set_title('Combined Supply Range', fontsize=12, fontweight='bold')
    ax9.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    plt.suptitle('High Voltage Supply Calibration Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Calibration plots saved to: {output_file}")
    plt.close()


# ============================================================================
# ORIGINAL CALIBRATION CLASSES AND FUNCTIONS
# ============================================================================

class SupplyCalibrationDMM:
    """Stores calibration data for a power supply (positive or negative) using DMM measurements."""

    def __init__(self, supply_name: str, dac_id: int, adc_channels: list[int]):
        self.supply_name = supply_name
        self.dac_id = dac_id
        self.adc_channels = adc_channels
        self.data = []
        self.coeffs = None  # Will store polynomial coefficients after generation

    def add_point(self, dac_value: int, raw_adc_ch0: int, raw_adc_ch1: int, dmm_voltage: float):
        """Add a calibration data point with DMM measurement."""
        self.data.append((dac_value, raw_adc_ch0, raw_adc_ch1, dmm_voltage))

    def find_dac_for_voltage(self, target_voltage: float) -> int:
        """Find DAC value for target voltage using calibration coefficients if available."""
        # Use polynomial coefficients if available
        if self.coeffs and 'volt_to_dac' in self.coeffs:
            dac_float = np.polyval(self.coeffs['volt_to_dac']['coeffs'], target_voltage)
            dac = int(np.round(dac_float))
            dac_min, dac_max = self.coeffs['dac_range']
            return max(dac_min, min(dac, dac_max))

        # Fallback to old method
        if not self.data:
            return 0
        closest = min(self.data, key=lambda x: abs(x[3] - target_voltage))
        return closest[0]

    def get_voltage_at_dac(self, dac_value: int) -> float:
        """Get the calibrated DMM voltage for a specific DAC value."""
        # Use polynomial coefficients if available
        if self.coeffs and 'dac_to_volt' in self.coeffs:
            return np.polyval(self.coeffs['dac_to_volt']['coeffs'], dac_value)

        # Fallback to interpolation
        for dac, raw_adc_ch0, raw_adc_ch1, dmm_voltage in self.data: # noqa: B007

            if dac == dac_value:
                return dmm_voltage

        if len(self.data) < 2:
            return 0.0

        sorted_data = sorted(self.data)
        for i in range(len(sorted_data) - 1):
            dac1, _, _, volt1 = sorted_data[i]
            dac2, _, _, volt2 = sorted_data[i + 1]
            if dac1 <= dac_value <= dac2:
                ratio = (dac_value - dac1) / (dac2 - dac1)
                return volt1 + ratio * (volt2 - volt1)

        if dac_value < sorted_data[0][0]:
            return sorted_data[0][3]
        return sorted_data[-1][3]

    def save_to_csv(self, filepath: Path):
        """Save calibration data to CSV file."""
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['DAC_Value', f'Raw_ADC_Ch{self.adc_channels[0]}', f'Raw_ADC_Ch{self.adc_channels[1]}', 'DMM_Voltage_V'])
            for dac_value, raw_adc_ch0, raw_adc_ch1, dmm_voltage in sorted(self.data):
                writer.writerow([dac_value, raw_adc_ch0, raw_adc_ch1, dmm_voltage])
        print(f"✅ Calibration data saved to {filepath}")

    def load_from_csv(self, filepath: Path):
        """Load calibration data from CSV file."""
        self.data.clear()
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dac_value = int(row['DAC_Value'])
                raw_adc_cols = [k for k in row.keys() if k.startswith('Raw_ADC_Ch')] # noqa: SIM118
                raw_adc_ch0 = int(row[raw_adc_cols[0]])
                raw_adc_ch1 = int(row[raw_adc_cols[1]])
                dmm_voltage = float(row['DMM_Voltage_V'])
                self.data.append((dac_value, raw_adc_ch0, raw_adc_ch1, dmm_voltage))
        print(f"✅ Loaded {len(self.data)} calibration points from {filepath}")


def display_voltages(interface):
    """Display current voltage monitor readings with calibrated values if available."""
    print("\nReading voltage monitor values...")
    voltages = interface.hvcontroller.get_vmon_values()

    print("\n" + "="*80)
    print("VOLTAGE MONITOR READINGS")
    print("="*80)

    # Check if calibration coefficients are available
    try:
        import calibration_coeffs as cal
        has_calibration = True
        print("✅ Using calibrated coefficients")
    except ImportError:
        has_calibration = False
        print("⚠️  No calibration coefficients found (run option 8 to generate)")

    print("="*80)

    for channel_data in voltages:
        ch = channel_data['channel']
        raw_adc = channel_data['raw_adc']
        voltage = channel_data['voltage']
        converted = channel_data['converted_voltage']

        # Calculate calibrated voltage if available
        calibrated_str = ""
        if has_calibration:
            try:
                if ch == 0:
                    cal_volt = cal.pos_adc_ch0_to_voltage(raw_adc)
                    calibrated_str = f"  |  Calibrated: {cal_volt:7.3f} V"
                elif ch == 1:
                    cal_volt = cal.pos_adc_ch1_to_voltage(raw_adc)
                    calibrated_str = f"  |  Calibrated: {cal_volt:7.3f} V"
                elif ch == 2:
                    cal_volt = cal.neg_adc_ch2_to_voltage(raw_adc)
                    calibrated_str = f"  |  Calibrated: {cal_volt:7.3f} V"
                elif ch == 3:
                    cal_volt = cal.neg_adc_ch3_to_voltage(raw_adc)
                    calibrated_str = f"  |  Calibrated: {cal_volt:7.3f} V"
            except Exception as e:
                calibrated_str = "  |  Calibrated: Error"

        print(f"Channel {ch}: "
              f"Raw ADC: {raw_adc:5d}  |  "
              f"Voltage: {voltage:6.3f} V  |  "
              f"Converted: {converted:6.3f} V{calibrated_str}")

    print("="*80)


def calibrate_supply(interface, calibration: SupplyCalibrationDMM, max_voltage: float = 65.0,
                     dac_step: int = 10, dmm_interval: int = 300):
    """Calibrate a power supply by stepping through DAC values with DMM measurements."""
    print(f"\n{'='*80}")
    print(f"CALIBRATING {calibration.supply_name.upper()} WITH DMM")
    print(f"{'='*80}")
    print(f"DAC ID: {calibration.dac_id}, ADC Channels: {calibration.adc_channels}")
    print(f"Max voltage limit: {max_voltage}V (using converted voltage as safety guide)")
    print(f"DAC step size: {dac_step}")
    print(f"DMM reading interval: every {dmm_interval} steps")
    print("\n⚠️  Ensure HV is ENABLED and DMM is connected before starting!")

    response = input("\nStart calibration? (y/n): ").strip().lower()
    if response != 'y':
        print("Calibration cancelled.")
        return

    calibration.data.clear()

    print("\nStarting calibration sweep...")
    print(f"{'DAC':>6} | {'RawADC0':>8} | {'RawADC1':>8} | {'ConvV':>8} | {'DMM (V)':>10} | {'Status':>10}")
    print("-" * 75)

    for dac_value in range(0, 4096, dac_step):
        interface.hvcontroller.set_raw_dac(dac_id=calibration.dac_id, dac_value=dac_value)
        time.sleep(0.05)

        voltages = interface.hvcontroller.get_vmon_values()
        ch0_data = voltages[calibration.adc_channels[0]]
        ch1_data = voltages[calibration.adc_channels[1]]
        raw_adc_ch0 = ch0_data['raw_adc']
        raw_adc_ch1 = ch1_data['raw_adc']
        voltage_ch0 = ch0_data['converted_voltage']
        voltage_ch1 = ch1_data['converted_voltage']

        max_measured = max(abs(voltage_ch0), abs(voltage_ch1))
        if max_measured >= max_voltage:
            print(f"{dac_value:6d} | {raw_adc_ch0:8d} | {raw_adc_ch1:8d} | {max_measured:7.2f}V | {'---':>10} | {'LIMIT':>10}")
            print(f"\n⚠️  Reached voltage limit ({max_voltage}V). Stopping calibration.")
            break

        if dac_value % dmm_interval == 0:
            print(f"\n{'='*75}")
            print(f"DAC: {dac_value}, Raw ADC Ch{calibration.adc_channels[0]}: {raw_adc_ch0}, Ch{calibration.adc_channels[1]}: {raw_adc_ch1}")
            print(f"Converted voltage (estimate): {max_measured:.2f}V")

            while True:
                dmm_reading = input("Enter DMM voltage reading (or 's' to skip, 'q' to quit): ").strip().lower()

                if dmm_reading == 'q':
                    print("Calibration stopped by user.")
                    interface.hvcontroller.set_raw_dac(dac_id=calibration.dac_id, dac_value=0)
                    return
                elif dmm_reading == 's':
                    print("Skipping this measurement point.")
                    break
                else:
                    try:
                        dmm_voltage = float(dmm_reading)
                        calibration.add_point(dac_value, raw_adc_ch0, raw_adc_ch1, dmm_voltage)
                        print(f"{dac_value:6d} | {raw_adc_ch0:8d} | {raw_adc_ch1:8d} | {max_measured:7.2f}V | {dmm_voltage:9.2f}V | {'RECORDED':>10}")
                        print(f"{'='*75}")
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric voltage value.")
        elif dac_value % (dac_step * 10) == 0 and dac_value % dmm_interval != 0:
            print(f"{dac_value:6d} | {raw_adc_ch0:8d} | {raw_adc_ch1:8d} | {max_measured:7.2f}V | {'---':>10} | {'OK':>10}")

    interface.hvcontroller.set_raw_dac(dac_id=calibration.dac_id, dac_value=0)

    print(f"\n✅ Calibration complete! Collected {len(calibration.data)} DMM data points.")
    if calibration.data:
        print(f"DAC range: {calibration.data[0][0]} to {calibration.data[-1][0]}")
        print(f"DMM voltage range: {calibration.data[0][3]:.2f}V to {calibration.data[-1][3]:.2f}V")


def set_hv_positive_voltage(interface, voltage, fine_tune=True):
    """Set the positive HV output voltage using calibration lookup."""
    try:
        import calibration_coeffs as cal
        has_calibration = True
    except ImportError:
        print("❌ No calibration coefficients found. Run option 8 to generate calibration first.")
        return False

    try:
        dac_value = cal.pos_voltage_to_dac(voltage)

        print(f"\n{'='*80}")
        print(f"SETTING POSITIVE HV OUTPUT: {voltage:.2f}V")
        print(f"{'='*80}")
        print("✅ Using calibrated coefficients")
        print(f"Calculated DAC value: {dac_value}")

        print("\nEnabling HV and setting target...")
        interface.hvcontroller.hv_enable(enable=True)
        time.sleep(0.1)

        # Set DAC directly using calibrated value
        interface.hvcontroller.set_raw_dac(dac_id=0, dac_value=dac_value)
        time.sleep(0.2)
        voltages = interface.hvcontroller.get_vmon_values()
        ch0_data = voltages[0]
        ch1_data = voltages[1]

        # Calculate calibrated readings
        cal_ch0 = cal.pos_adc_ch0_to_voltage(ch0_data['raw_adc'])
        cal_ch1 = cal.pos_adc_ch1_to_voltage(ch1_data['raw_adc'])

        print("\n✅ Target reached!")
        print(f"   DAC: {dac_value}")
        print(f"   Target voltage: {voltage:.2f}V")
        print(f"   Ch0 - Raw ADC: {ch0_data['raw_adc']:5d} | Calibrated: {cal_ch0:7.3f}V")
        print(f"   Ch1 - Raw ADC: {ch1_data['raw_adc']:5d} | Calibrated: {cal_ch1:7.3f}V")

        if not fine_tune:
            return True

        print(f"\n{'='*80}")
        print("FINE-TUNING MODE")
        print(f"{'='*80}")
        print("Commands: +N (increase DAC by N), -N (decrease DAC by N), q (quit)")

        while True:
            cmd = input("\nAdjustment: ").strip().lower()

            if cmd == 'q':
                break

            try:
                adjustment = int(cmd)
                dac_value += adjustment
                dac_value = max(0, min(4095, dac_value))

                interface.hvcontroller.set_raw_dac(dac_id=0, dac_value=dac_value)
                time.sleep(0.05)

                voltages = interface.hvcontroller.get_vmon_values()
                ch0_data = voltages[0]
                ch1_data = voltages[1]

                cal_ch0 = cal.pos_adc_ch0_to_voltage(ch0_data['raw_adc'])
                cal_ch1 = cal.pos_adc_ch1_to_voltage(ch1_data['raw_adc'])

                print(f"   DAC: {dac_value}")
                print(f"   Ch0 - Raw ADC: {ch0_data['raw_adc']:5d} | Calibrated: {cal_ch0:7.3f}V")
                print(f"   Ch1 - Raw ADC: {ch1_data['raw_adc']:5d} | Calibrated: {cal_ch1:7.3f}V")
            except ValueError:
                print("Invalid command. Use +N, -N, or q")

        return True
    except Exception as e:
        print(f"❌ Error setting positive HV voltage: {e}")
        return False


def set_hv_negative_voltage(interface, voltage, fine_tune=True):
    """Set the negative HV output voltage using calibration lookup."""
    try:
        import calibration_coeffs as cal
        has_calibration = True
    except ImportError:
        print("❌ No calibration coefficients found. Run option 8 to generate calibration first.")
        return False

    try:
        dac_value = cal.neg_voltage_to_dac(voltage)

        print(f"\n{'='*80}")
        print(f"SETTING NEGATIVE HV OUTPUT: {voltage:.2f}V")
        print(f"{'='*80}")
        print("✅ Using calibrated coefficients")
        print(f"Calculated DAC value: {dac_value}")

        print("\nEnabling HV and setting target...")
        interface.hvcontroller.hv_enable(enable=True)
        time.sleep(0.1)

        # Set DAC directly using calibrated value
        interface.hvcontroller.set_raw_dac(dac_id=1, dac_value=dac_value)
        time.sleep(0.2)
        voltages = interface.hvcontroller.get_vmon_values()
        ch2_data = voltages[2]
        ch3_data = voltages[3]

        # Calculate calibrated readings
        cal_ch2 = cal.neg_adc_ch2_to_voltage(ch2_data['raw_adc'])
        cal_ch3 = cal.neg_adc_ch3_to_voltage(ch3_data['raw_adc'])

        print("\n✅ Target reached!")
        print(f"   DAC: {dac_value}")
        print(f"   Target voltage: {voltage:.2f}V")
        print(f"   Ch2 - Raw ADC: {ch2_data['raw_adc']:5d} | Calibrated: {cal_ch2:7.3f}V")
        print(f"   Ch3 - Raw ADC: {ch3_data['raw_adc']:5d} | Calibrated: {cal_ch3:7.3f}V")

        if not fine_tune:
            return True

        print(f"\n{'='*80}")
        print("FINE-TUNING MODE")
        print(f"{'='*80}")
        print("Commands: +N (increase DAC by N), -N (decrease DAC by N), q (quit)")

        while True:
            cmd = input("\nAdjustment: ").strip().lower()

            if cmd == 'q':
                break

            try:
                adjustment = int(cmd)
                dac_value += adjustment
                dac_value = max(0, min(4095, dac_value))

                interface.hvcontroller.set_raw_dac(dac_id=1, dac_value=dac_value)
                time.sleep(0.05)

                voltages = interface.hvcontroller.get_vmon_values()
                ch2_data = voltages[2]
                ch3_data = voltages[3]

                cal_ch2 = cal.neg_adc_ch2_to_voltage(ch2_data['raw_adc'])
                cal_ch3 = cal.neg_adc_ch3_to_voltage(ch3_data['raw_adc'])

                print(f"   DAC: {dac_value}")
                print(f"   Ch2 - Raw ADC: {ch2_data['raw_adc']:5d} | Calibrated: {cal_ch2:7.3f}V")
                print(f"   Ch3 - Raw ADC: {ch3_data['raw_adc']:5d} | Calibrated: {cal_ch3:7.3f}V")
            except ValueError:
                print("Invalid command. Use +N, -N, or q")

        return True
    except Exception as e:
        print(f"❌ Error setting negative HV voltage: {e}")
        return False


def display_menu():
    """Display the main menu."""
    print("\n" + "="*80)
    print("HIGH VOLTAGE SUPPLY CALIBRATION TOOL (DMM)")
    print("="*80)
    print("1. Display voltage monitor readings")
    print("2. Calibrate positive supply with DMM")
    print("3. Calibrate negative supply with DMM")
    print("4. Set target voltage (positive supply)")
    print("5. Set target voltage (negative supply)")
    print("6. Save calibration data to CSV")
    print("7. Load calibration data from CSV")
    print("8. Generate calibration coefficients (Python + C)")
    print("9. Plot calibration data")
    print("10. Enable HV output")
    print("11. Disable HV output")
    print("0. Exit")
    print("="*80)


def main():
    print("Starting LIFU Supply Calibration Tool...")
    interface = LIFUInterface()
    tx_connected, hv_connected = interface.is_device_connected()

    if not hv_connected:
        print("❌ LIFU Console not connected.")
        sys.exit(1)

    print("✅ LIFU Console connected.")

    pos_cal = SupplyCalibrationDMM("Positive Supply", dac_id=0, adc_channels=[0, 1])
    neg_cal = SupplyCalibrationDMM("Negative Supply", dac_id=1, adc_channels=[2, 3])

    print("\nDisabling HV output for safety...")
    interface.hvcontroller.hv_enable(enable=False)

    print("Initializing all DACs to 0V...")
    interface.hvcontroller.set_raw_dac(dac_id=0, dac_value=0)
    interface.hvcontroller.set_raw_dac(dac_id=1, dac_value=0)

    while True:
        display_menu()
        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("\nExiting...")
            interface.hvcontroller.hv_enable(enable=False)
            interface.hvcontroller.set_raw_dac(dac_id=0, dac_value=0)
            interface.hvcontroller.set_raw_dac(dac_id=1, dac_value=0)
            print("HV disabled and all DACs reset to 0V")
            break

        elif choice == "1": # noqa: RET508
            display_voltages(interface)

        elif choice == "2":
            try:
                max_v = float(input("Enter max voltage limit (default 65V, max 100V): ") or "65")
                max_v = min(max_v, 100.0)
                step = int(input("Enter DAC step size (default 10): ") or "10")
                dmm_int = int(input("Enter DMM reading interval in steps (default 300): ") or "300")
                calibrate_supply(interface, pos_cal, max_voltage=max_v, dac_step=step, dmm_interval=dmm_int)
            except ValueError:
                print("Invalid input.")

        elif choice == "3":
            try:
                max_v = float(input("Enter NEGATIVE voltage limit magnitude (default -65V, min -100V): ") or "-65")
                # Ensure it is negative and not less than -100V
                if max_v > 0:
                    max_v = -abs(max_v)
                if max_v > -65.0:
                    # Closer to zero than -65 -> clamp to -65
                    max_v = -65.0
                if max_v < -100.0:
                    max_v = -100.0

                step = int(input("Enter DAC step size (default 10): ") or "10")
                dmm_int = int(input("Enter DMM reading interval in steps (default 300): ") or "300")

                # Pass absolute value into calibrate_supply for safety check,
                # since converted voltages for negative rail will already be negative
                calibrate_supply(interface, neg_cal, max_voltage=abs(max_v), dac_step=step, dmm_interval=dmm_int)
            except ValueError:
                print("Invalid input.")

        elif choice == "4":
            try:
                target = float(input("Enter target voltage for positive supply (V): "))
                set_hv_positive_voltage(interface, target)
            except ValueError:
                print("Invalid input.")

        elif choice == "5":
            try:
                target = float(input("Enter target voltage for negative supply (V, use negative values): "))
                set_hv_negative_voltage(interface, target)
            except ValueError:
                print("Invalid input.")

        elif choice == "6":
            try:
                current_dir = Path.cwd()
                pos_file = current_dir / "pos_cal.csv"
                neg_file = current_dir / "neg_cal.csv"

                if pos_cal.data:
                    pos_cal.save_to_csv(pos_file)
                if neg_cal.data:
                    neg_cal.save_to_csv(neg_file)
            except Exception as e:
                print(f"Error saving: {e}")

        elif choice == "7":
            try:
                current_dir = Path.cwd()
                pos_file = current_dir / "pos_cal.csv"
                neg_file = current_dir / "neg_cal.csv"

                if pos_file.exists():
                    pos_cal.load_from_csv(pos_file)
                if neg_file.exists():
                    neg_cal.load_from_csv(neg_file)
            except Exception as e:
                print(f"Error loading: {e}")

        elif choice == "8":
            print("\n" + "="*80)
            print("GENERATING CALIBRATION COEFFICIENTS")
            print("="*80)

            current_dir = Path.cwd()
            pos_file = current_dir / "pos_cal.csv"
            neg_file = current_dir / "neg_cal.csv"

            # Generate coefficients
            pos_results, neg_results = generate_calibration_coefficients(
                pos_csv_path=pos_file if pos_file.exists() else None,
                neg_csv_path=neg_file if neg_file.exists() else None
            )

            if pos_results or neg_results:
                # Save to files
                save_python_coefficients(pos_results, neg_results, 'calibration_coeffs.py')
                save_c_header(pos_results, neg_results, 'hv_calibration_coeffs.h')
                save_c_implementation(pos_results, neg_results, 'hv_calibration_coeffs.c')

                # Store coefficients in calibration objects
                if pos_results:
                    pos_cal.coeffs = pos_results
                if neg_results:
                    neg_cal.coeffs = neg_results

                print("\n✅ Calibration coefficients generated successfully!")
                print("   - calibration_coeffs.py (Python)")
                print("   - hv_calibration_coeffs.h (C header)")
                print("   - hv_calibration_coeffs.c (C implementation)")
            else:
                print("❌ No calibration CSV files found. Run calibration first.")

        elif choice == "9":
            print("\n" + "="*80)
            print("PLOTTING CALIBRATION DATA")
            print("="*80)

            current_dir = Path.cwd()
            pos_file = current_dir / "pos_cal.csv"
            neg_file = current_dir / "neg_cal.csv"

            plot_calibration_data(
                pos_csv_path=pos_file if pos_file.exists() else None,
                neg_csv_path=neg_file if neg_file.exists() else None,
                output_file='calibration_plots.png'
            )

        elif choice == "10":
            print("\nEnabling HV output...")
            if interface.hvcontroller.hv_enable(enable=True):
                print("✅ HV output enabled successfully")
            else:
                print("❌ Failed to enable HV output")

        elif choice == "11":
            print("\nDisabling HV output...")
            if interface.hvcontroller.hv_enable(enable=False):
                print("✅ HV output disabled successfully")
            else:
                print("❌ Failed to disable HV output")

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
