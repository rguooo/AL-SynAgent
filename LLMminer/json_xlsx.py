import os
import json
import pandas as pd

root_dir = r"your-file-path"

json_files = []
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    auto_dir = os.path.join(folder_path, "auto")
    if os.path.isdir(auto_dir):
        for file in os.listdir(auto_dir):
            if file.endswith("_ferroelectric_output.json"):
                json_files.append(os.path.join(auto_dir, file))

print(f"Found {len(json_files)} JSON files")

def safe_get(data, keys, default=None):
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result if result is not None else default

def extract_property_list(props):
    if not isinstance(props, list):
        return ""
    result = []
    for prop in props:
        if isinstance(prop, str):
            result.append(prop)
        elif isinstance(prop, dict):
            p = prop.get("property", "").strip()
            v = prop.get("value", "").strip()
            u = prop.get("unit", "").strip()
            if p and v and u:
                result.append(f"{p}: {v} {u}")
            elif p and v:
                result.append(f"{p}: {v}")
            elif p:
                result.append(p)
    return ", ".join(result)

def extract_data_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON parsing failed: {file_path}, error: {e}")
        return {}

    extracted = {
        "Material Name": safe_get(data, ["meta", "material_name"]),
        "Chemical Formula": safe_get(data, ["meta", "chemical_formula"]),
        "Material Type": safe_get(data, ["meta", "material_type"]),
        "Crystal Structure": safe_get(data, ["meta", "crystal_structure"]),
        "Space Group": safe_get(data, ["meta", "space_group"]),
        "Band Gap (eV)": safe_get(data, ["meta", "band gap"]),
        "Crystal Size": safe_get(data, ["meta", "crystal_size"]),
        "CCDC Number": safe_get(data, ["meta", "ccdc_number"]),
        "curie_temperature_value": safe_get(data, ["curie_temperature", "value"]),
        "curie_temperature_unit": safe_get(data, ["curie_temperature", "unit"]),
        "curie_temperature_method": safe_get(data, ["curie_temperature", "measurement_method"]),
        "curie_temperature_notes": safe_get(data, ["curie_temperature", "notes"]),
        "spontaneous_polarization_value": safe_get(data, ["polarization", "spontaneous_polarization", "value"]),
        "spontaneous_polarization_unit": safe_get(data, ["polarization", "spontaneous_polarization", "unit"]),
        "spontaneous_polarization_conditions": safe_get(data, ["polarization", "spontaneous_polarization", "measurement_conditions"]),
        "saturated_polarization_value": safe_get(data, ["polarization", "saturated_polarization", "value"]),
        "saturated_polarization_unit": safe_get(data, ["polarization", "saturated_polarization", "unit"]),
        "saturated_polarization_conditions": safe_get(data, ["polarization", "saturated_polarization", "measurement_conditions"]),
        "coercive_field_value": safe_get(data, ["polarization", "coercive_field", "value"]),
        "coercive_field_unit": safe_get(data, ["polarization", "coercive_field", "unit"]),
        "remnant_polarization_value": safe_get(data, ["polarization", "remnant_polarization", "value"]),
        "remnant_polarization_unit": safe_get(data, ["polarization", "remnant_polarization", "unit"]),
        "remnant_polarization_conditions": safe_get(data, ["polarization", "remnant_polarization", "measurement_conditions"]),
        "ec_value": safe_get(data, ["polarization", "ec", "value"]),
        "ec_unit": safe_get(data, ["polarization", "ec", "unit"]),
        "Leakage Current (A/cm²)": safe_get(data, ["polarization", "leakage_current"]),
        "Polarization Measurement Method": safe_get(data, ["polarization", "measurement_method"]),
        "Polarization Notes": safe_get(data, ["polarization", "notes"]),
        "Synthesis Precursors": "",
        "Synthesis Solvents": "",
        "Synthesis Method": safe_get(data, ["synthesis_information", "method"]),
        "Synthesis Temperature_value": safe_get(data, ["synthesis_information", "temperature", "value"]),
        "Synthesis Temperature_unit": safe_get(data, ["synthesis_information", "temperature", "unit"]),
        "Synthesis Time_value": safe_get(data, ["synthesis_information", "time", "value"]),
        "Synthesis Time_unit": safe_get(data, ["synthesis_information", "time", "unit"]),
        "atmosphere": safe_get(data, ["synthesis_information", "atmosphere"]),
        "heating_rate": safe_get(data, ["synthesis_information", "heating_rate"]),
        "Cooling Condition": safe_get(data, ["synthesis_information", "cooling_condition"]),
        "Substrate": safe_get(data, ["synthesis_information", "substrate"]),
        "Post Treatment": safe_get(data, ["synthesis_information", "post_treatment"]),
        "equipment": safe_get(data, ["synthesis_information", "equipment"]),
        "Additional Steps": "",
        "yield": safe_get(data, ["synthesis_information", "yield"]),
        "Dielectric Constant_value": safe_get(data, ["other_properties", "dielectric_constant", "value"]),
        "Dielectric Constant_unit": safe_get(data, ["other_properties", "dielectric_constant", "unit"]),
        "Dielectric Constant_frequency": safe_get(data, ["other_properties", "dielectric_constant", "frequency"]),
        "piezoelectric_coefficient_value": safe_get(data, ["other_properties", "piezoelectric_coefficient", "value"]),
        "piezoelectric_coefficient_unit": safe_get(data, ["other_properties", "piezoelectric_coefficient", "unit"]),
        "resistivity_value": safe_get(data, ["other_properties", "resistivity", "value"]),
        "resistivity_unit": safe_get(data, ["other_properties", "resistivity", "unit"]),
        "conductivity_value": safe_get(data, ["other_properties", "conductivity", "value"]),
        "conductivity_unit": safe_get(data, ["other_properties", "conductivity", "unit"]),
        "moderate_stability": safe_get(data, ["other_properties", "moderate_stability"]),
        "absorption_coefficient_value": safe_get(data, ["other_properties", "absorption_coefficient", "value"]),
        "absorption_coefficient_unit": safe_get(data, ["other_properties", "absorption_coefficient", "unit"]),
        "absorption_wavelength_value": safe_get(data, ["other_properties", "absorption_wavelength", "value"]),
        "absorption_wavelength_unit": safe_get(data, ["other_properties", "absorption_wavelength", "unit"]),
        "extinction_coefficient_value": safe_get(data, ["other_properties", "extinction_coefficient", "value"]),
        "extinction_coefficient_unit": safe_get(data, ["other_properties", "extinction_coefficient", "unit"]),
        "largest_diff_peak_and_hole_value": safe_get(data, ["other_properties", "largest_diff_peak_and_hole", "value"]),
        "largest_diff_peak_and_hole_unit": safe_get(data, ["other_properties", "largest_diff_peak_and_hole", "unit"]),
        "breakdown_field_value": safe_get(data, ["other_properties", "breakdown_field", "value"]),
        "breakdown_field_unit": safe_get(data, ["other_properties", "breakdown_field", "unit"]),
        "elastic_properties": extract_property_list(safe_get(data, ["other_properties", "elastic_properties"], [])),
        "optical_properties": extract_property_list(safe_get(data, ["other_properties", "optical_properties"], [])),
        "thermal_properties": extract_property_list(safe_get(data, ["other_properties", "thermal_properties"], [])),
        "photovoltaic_properties_zero_bias_photocurrent_density_value": safe_get(data, ["photovoltaic_properties", "zero_bias_photocurrent_density", "value"]),
        "photovoltaic_properties_zero_bias_photocurrent_density_unit": safe_get(data, ["photovoltaic_properties", "zero_bias_photocurrent_density", "unit"]),
        "photovoltaic_properties_on_off_switching_ratio_value": safe_get(data, ["photovoltaic_properties", "on_off_switching_ratio", "value"]),
        "photovoltaic_properties_on_off_switching_ratio_unit": safe_get(data, ["photovoltaic_properties", "on_off_switching_ratio", "unit"]),
        "photovoltaic_properties_wavelength_value": safe_get(data, ["photovoltaic_properties", "wavelength", "value"]),
        "photovoltaic_properties_wavelength_unit": safe_get(data, ["photovoltaic_properties", "wavelength", "unit"]),
        "photovoltaic_properties_lifetime_value": safe_get(data, ["photovoltaic_properties", "lifetime", "value"]),
        "photovoltaic_properties_lifetime_unit": safe_get(data, ["photovoltaic_properties", "lifetime", "unit"]),
        "photovoltaic_properties_pce_value": safe_get(data, ["photovoltaic_properties", "pce", "value"]),
        "photovoltaic_properties_pce_unit": safe_get(data, ["photovoltaic_properties", "pce", "unit"]),
        "photovoltaic_properties_quantum_yield_value": safe_get(data, ["photovoltaic_properties", "quantum_yield", "value"]),
        "photovoltaic_properties_quantum_yield_unit": safe_get(data, ["photovoltaic_properties", "quantum_yield", "unit"]),
        "photovoltaic_properties_open_circuit_voltage_value": safe_get(data, ["photovoltaic_properties", "open_circuit_voltage", "value"]),
        "photovoltaic_properties_open_circuit_voltage_unit": safe_get(data, ["photovoltaic_properties", "open_circuit_voltage", "unit"]),
        "photovoltaic_properties_short_circuit_current_value": safe_get(data, ["photovoltaic_properties", "short_circuit_current", "value"]),
        "photovoltaic_properties_short_circuit_current_unit": safe_get(data, ["photovoltaic_properties", "short_circuit_current", "unit"]),
        "photovoltaic_properties_rise_time_value": safe_get(data, ["photovoltaic_properties", "rise_time", "value"]),
        "photovoltaic_properties_rise_time_unit": safe_get(data, ["photovoltaic_properties", "rise_time", "unit"]),
        "photovoltaic_properties_decay_time_value": safe_get(data, ["photovoltaic_properties", "decay_time", "value"]),
        "photovoltaic_properties_decay_time_unit": safe_get(data, ["photovoltaic_properties", "decay_time", "unit"]),
        "Switching Cycles": safe_get(data, ["fatigue_testing", "switching_cycles"]),
        "Applications": ", ".join(safe_get(data, ["applications"], [])),
    }

    synthesis_info = safe_get(data, ["synthesis_information"], {})
    precursors = safe_get(synthesis_info, ["precursors"], [])
    if not isinstance(precursors, list):
        precursors = []
    precursor_names = [p["name"] for p in precursors if isinstance(p, dict)]
    extracted["Synthesis Precursors"] = ", ".join(precursor_names)
    solvents = safe_get(synthesis_info, ["solvents"], [])
    if not isinstance(solvents, list):
        solvents = []
    solvent_names = [s["name"] for s in solvents if isinstance(s, dict)]
    extracted["Synthesis Solvents"] = ", ".join(solvent_names)
    steps = safe_get(synthesis_info, ["additional_steps"], [])
    if not isinstance(steps, list):
        steps = []
    extracted["Additional Steps"] = ", ".join(steps)

    return extracted

all_data = []
for file in json_files:
    try:
        filename = os.path.basename(file)
        file_id = filename.split("_")[0]
        
        data = extract_data_from_json(file)
        data["ID"] = file_id
        all_data.append(data)
    except Exception as e:
        print(f"File reading failed: {file}, error: {e}")

df = pd.DataFrame(all_data)

output_excel_path = os.path.join(root_dir, "ferroelectric_materials.xlsx")
df.to_excel(output_excel_path, index=False, engine='openpyxl')
print(f"Data saved to Excel file: {output_excel_path}")