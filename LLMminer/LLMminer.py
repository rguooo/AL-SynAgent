import os
import json
import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("openai-api-key")

headers = {
    "Content-Type": "application/json",
    "Authorization": API_KEY
}

model = "chatgpt-4o-latest"

def call_linkai(prompt, temperature=0.2, retries=3):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are a precise extractor for materials science literature, specializing in ferroelectric and piezoelectric materials. AppCode: {APP_CODE}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            else:
                print(f"Unexpected response structure: {result}")
        except Exception as e:
            print(f"[Retry {attempt+1}] API call failed: {e}")
            time.sleep(2)
    return ""

def extract_structured_info(content):
    prompt = f"""You are an expert assistant trained to extract structured data from materials science documents, especially focusing on ferroelectric and piezoelectric materials.

You are given content from a scientific paper, and your task is to extract and fill the following JSON template. Pay special attention to Curie temperature, polarization properties, and synthesis information. If a field is not mentioned in the text, use an empty string or empty list as appropriate. Be precise and concise.

Target format:
{{
  "meta": {{
    "material_name": "",
    "chemical_formula": "",
    "material_type": "",
    "crystal_structure": "",
    "space_group": "",
    "band gap": "",
    "crystal_size": "",
    "ccdc_number": ""    
  }},
  "curie_temperature": {{
    "value": "",
    "unit": "",
    "measurement_method": "",
    "notes": ""
  }},
  "polarization": {{
    "spontaneous_polarization": {{
      "value": "",
      "unit": "",
      "measurement_conditions": ""
    }},
    "saturated_polarization": {{
      "value": "",
      "unit": "",
      "measurement_conditions": ""
    }},
    "coercive_field": {{
      "value": "",
      "unit": ""
    }},
    "remnant_polarization": {{
      "value": "",
      "unit": "",
      "measurement_conditions": ""
    }},
    "ec": {{
      "value": "",
      "unit": ""
    }},
    "leakage_current": "",
    "measurement_method": "",
    "notes": ""
  }},
  "synthesis_information": {{
    "precursors": [
      {{
        "name": "",
        "amount": "",
        "unit": "",
        "purity": ""
      }}
    ],
    "solvents": [
      {{
        "name": "",
        "amount": "",
        "unit": ""
      }}
    ],
    "method": "",
    "temperature": {{
      "value": "",
      "unit": ""
    }},
    "time": {{
      "value": "",
      "unit": ""
    }},
    "atmosphere": "",
    "heating_rate": "",
    "cooling_condition": "",
    "substrate": "",
    "post_treatment": "",
    "equipment": "",
    "additional_steps": [],
    "yield": ""
  }},
  "other_properties": {{
    "dielectric_constant": {{
      "value": "",
      "unit": "",
      "frequency": ""
    }},
    "piezoelectric_coefficient": {{
      "value": "",
      "unit": "",
    }},
    "resistivity": {{
      "value": "",
      "unit": ""
    }},
    "conductivity": {{
      "value": "",
      "unit": ""
    }},
    "moderate_stability": "",
    "absorption_coefficient": {{
      "value": "",
      "unit": ""
    }},
    "absorption_wavelength": {{
      "value": "",
      "unit": ""
    }},
    "extinction_coefficient": {{
      "value": "",
      "unit": ""
    }},
    "largest_diff_peak_and_hole": {{
      "value": "",
      "unit": ""
    }},
    "breakdown_field": {{
      "value": "",
      "unit": ""
    }},
    "elastic_properties": [],
    "optical_properties": [],
    "thermal_properties": []
  }},
    "photovoltaic_properties": {{
    "zero_bias_photocurrent_density": {{
      "value": "",
      "unit": ""
    }},
    "on_off_switching_ratio": {{
      "value": "",
      "unit": ""
    }},
    "wavelength": {{
      "value": "",
      "unit": ""
    }},
    "lifetime": {{
      "value": "",
      "unit": ""
    }},
    "pce": {{
      "value": "",
      "unit": ""
    }},
    "quantum_yield": {{
      "value": "",
      "unit": ""
    }},
    "open_circuit_voltage": {{
      "value": "",
      "unit": ""
    }},
    "short_circuit_current": {{
      "value": "",
      "unit": ""
    }},
    "rise_time": {{
      "value": "",
      "unit": ""
    }},
    "decay_time": {{
      "value": "",
      "unit": ""
    }}
  }},
  "fatigue_testing": {{
    "switching_cycles": "",
    "frequency": {{
      "value": "",
      "unit": ""
    }}
  }},
  "characterization": {{
    "techniques_used": [],
    "equipment_details": []
  }},
  "applications": [],
  "dataset_source": "",
  "synthesis_factors": {{
    "metal_electronegativity": "",
    "metal_ionization_energy": "",
    "molecule_electronegativity": "",
    "molecule_functional_groups": [],
    "molecule_dipole_moment": "",
    "molecule_hardness": "",
    "molecule_homo": "",
    "molecule_lumo": "",
    "molecule_homo_lumo_gap": "",
    "hydrogen_bond_strength": "",
    "organic_organic_interaction_strength": "",
    "organic_inorganic_interaction_strength": "",
    "molecule_rotational_inertia": "",
    "molecule_free_rotation": "",
    "molecular polarization",
    "molecular rigidity"
    "notes": ""
}}
}}

Instructions for extraction:
1. For Curie temperature, look for terms like: "Curie temperature", "Tc", "transition temperature", "transition point", "ferroelectric transition", "phase transition temperature". Units are typically K or °C.
2. For polarization, look for: "spontaneous polarization", "Ps", "saturated polarization", "remnant polarization", "Pr". Units are typically μC/cm², μC·cm⁻², C/m², or similar.
3. For synthesis information, focus on Method/Experimental/Synthesis sections. Extract specific amounts, temperatures, times, and solutions. look for terms like: "molar ratio", "grinding", "calcinated","calcined","sintered", "dried", "stirred", "heated". Units are typically °C, h , mL, g, mg, L, mol, mmol.
4. Pay attention to numerical values with their units and measurement conditions.
5. For crystal size, look for terms like: "crystal size", "grain size", "particle size", "dimensions".
6. For synthesis yield, look for: "yield", "synthesis yield", "product yield".
7. For resistivity, look for: "resistivity", "ρ", "electrical resistivity".
8. For conductivity, look for: "conductivity", "σ", "electrical conductivity".
9. For moderate stability, check: "stability under humidity", "temperature stability", "environmental stability", "high stability". Units are typically days, weeks, months, hours.
10. For CCDC number, find: "CCDC", "Cambridge Crystallographic Data Centre", "deposition number".
11. For zero-bias photocurrent density, search: "zero-bias photocurrent density", "Jph", "photocurrent density".
12. For on/off switching ratio, use keywords: "on/off ratio", "switching ratio", "current on/off ratio".
13. For wavelength, extract:"photoluminescence peak", "λ", "wavelength", "incident light wavelength".
14. For lifetime, check: "carrier lifetime", "exciton lifetime", "device lifetime".
15. For PCE, look for: "power conversion efficiency", "PCE", "η".
16. For quantum yield, find: "quantum yield", "QY", "external quantum efficiency".
17. For Voc/Isc, use: "open-circuit voltage", "Voc", "short-circuit current", "Isc".
18. For absorption coefficient/wavelength, extract: "absorption coefficient", "α", "absorption spectrum".
19. For piezoelectric coefficient, see existing fields.
20. For breakdown field, check: "breakdown field", "electric breakdown", "dielectric breakdown".
21. For fatigue testing, extract: "fatigue test", "switching cycles", "number of cycles", "frequency".
22. For extinction coefficient, use: "extinction coefficient", "ε", "molar extinction coefficient".
23. For largest diff. peak and hole, find: "diffraction peak", "electron density", "peak and hole".
24. For rise/decay time, look for: "rise time", "decay time", "response time".
25. For remnant polarization (Pr), extract: "remnant polarization", "Pr", "remanent polarization".
26. For coercive field (Ec), see: "coercive field", "Ec", "coercive voltage".
27. For leakage current, extract: "leakage current", "IL", "current leakage".
28. For synthesis factors, extract the following:
29. metal electronegativity: look for terms like "electronegativity", "X value", "Pauling scale"
30. metal ionization energy: search for "ionization energy", "IE", "first ionization potential"
31. molecule electronegativity: find "molecular electronegativity", "χ value"
32. molecule functional groups: extract any functional group mentioned (e.g., -OH, -NH2, -COOH)
33. molecule dipole moment: check for "dipole moment", "μ", "Debye unit"
34. molecule chemical hardness: look for "chemical hardness", "η", "global hardness"
35. molecule HOMO-LUMO gap: extract "HOMO-LUMO gap", "bandgap", "energy gap"
36. hydrogen bond strength: note phrases like "hydrogen bonding", "H-bond strength", "interaction with inorganic framework"
37. organic-organic interaction strength: find "π–π stacking", "van der Waals interaction", "organic–organic interaction"
38. molecule rotational inertia: mention of "rotational inertia", "moment of inertia"
39. molecule free rotation: check if the text describes "free rotation", "restricted rotation", "molecular flexibility".


Content:
\"\"\"{content}\"\"\"

Only return the filled JSON. Do not include any explanations or additional text."""
    
    raw = call_linkai(prompt)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].strip()
    
    return raw

def process_markdown_file(md_path):
    out_path = md_path.replace(".md", "_ferroelectric_output.json")
    if not os.path.exists(md_path):
        print(f"File not found: {md_path}")
        return False

    print(f"\n Processing: {md_path}")
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {md_path}: {e}")
        return False

    print("Extracting ferroelectric material information...")
    extracted = extract_structured_info(content)
    
    if not extracted:
        print("Empty result returned")
        return False
    
    try:
        parsed = json.loads(extracted)
        curie_temp = parsed.get("curie_temperature", {}).get("value", "")
        polarization = parsed.get("polarization", {})
        synthesis = parsed.get("synthesis_information", {})
        
        print("Extraction Summary:")
        if curie_temp:
            print(f"Curie Temperature: {curie_temp} {parsed.get('curie_temperature', {}).get('unit', '')}")
        else:
            print("Curie Temperature: Not found")
            
        spont_pol = polarization.get("spontaneous_polarization", {}).get("value", "")
        sat_pol = polarization.get("saturated_polarization", {}).get("value", "")
        if spont_pol or sat_pol:
            print(f"   • Spontaneous Polarization: {spont_pol} {polarization.get('spontaneous_polarization', {}).get('unit', '')}")
            print(f"   • Saturated Polarization: {sat_pol} {polarization.get('saturated_polarization', {}).get('unit', '')}")
        else:
            print("   • Polarization: Not found")
            
        if synthesis.get("method") or synthesis.get("precursors"):
            print(f"   • Synthesis Method: {synthesis.get('method', 'Found')}")
            print(f"   • Number of Precursors: {len(synthesis.get('precursors', []))}")
        else:
            print("   • Synthesis Information: Not found")
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"Successfully extracted information and saved to: {out_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        error_path = md_path.replace(".md", "_error.txt")
        with open(error_path, "w", encoding="utf-8") as errf:
            errf.write(f"File: {md_path}\n\n")
            errf.write(f"JSON Parsing Error: {e}\n\n")
            errf.write(f"Raw Response:\n{extracted}\n")
        print(f"Error details saved to: {error_path}")
        return False

def get_available_file_indices(base_dir):
    available_indices = [] 
    if not os.path.exists(base_dir):
        print(f" Base directory does not exist: {base_dir}")
        return available_indices
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            md_path = os.path.join(item_path, "auto", f"{item}.md")
            if os.path.exists(md_path):
                available_indices.append(int(item))
    
    return sorted(available_indices)

def batch_process_files(base_dir, start_idx=None, end_idx=None):

    available_indices = get_available_file_indices(base_dir)
    
    if not available_indices:
        print("No available files found to process")
        return
    
    print(f"Found available files: {available_indices}")

    if start_idx is None:
        start_idx = min(available_indices)

    if end_idx is None:
        end_idx = max(available_indices)

    files_to_process = [idx for idx in available_indices if start_idx <= idx <= end_idx]
    
    if not files_to_process:
        print(f"No files found in the specified range [{start_idx}, {end_idx}]")
        print(f"Available files: {available_indices}")
        return
    
    print(f"Starting batch processing for files: {files_to_process}")
    print(f"Range: {min(files_to_process)} to {max(files_to_process)}")
    
    success_count = 0
    error_count = 0
    processed_files = []
    
    for idx in files_to_process:
        md_path = os.path.join(base_dir, f"{idx:03d}", "auto", f"{idx:03d}.md")
        
        print(f"\n{'='*50}")
        print(f"Processing file {idx:03d} ({files_to_process.index(idx)+1}/{len(files_to_process)})")
        
        try:
            if process_markdown_file(md_path):
                success_count += 1
                processed_files.append(f"{idx:03d}")
            else:
                error_count += 1
        except Exception as e:
            print(f"Unexpected error processing {md_path}: {e}")
            error_count += 1
        
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f" Batch Processing Summary:")
    print(f"  • Successfully processed: {success_count}")
    print(f"  • Errors encountered: {error_count}")
    print(f"  • Total files attempted: {len(files_to_process)}")
    print(f"  • Success rate: {success_count/len(files_to_process)*100:.1f}%")
    
    if processed_files:
        print(f"Successfully processed files: {', '.join(processed_files)}")

def scan_available_files(base_dir):
    """扫描可用的md文件"""
    print(f"Scanning directory: {base_dir}")
    available_files = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return available_files
    
    available_indices = get_available_file_indices(base_dir)
    
    for idx in available_indices:
        item = f"{idx:03d}"
        md_path = os.path.join(base_dir, item, "auto", f"{item}.md")
        file_size = os.path.getsize(md_path) / 1024  # KB
        available_files.append((item, md_path, file_size))
        print(f"Found: {item}.md ({file_size:.1f} KB)")
    
    print(f"\n Total available files: {len(available_files)}")
    if available_files:
        indices = [int(f[0]) for f in available_files]
        print(f" File range: {min(indices)} to {max(indices)}")
    return available_files

if __name__ == "__main__":
    base_dir = r"your-file-path"
    available_files = scan_available_files(base_dir)  
    if available_files:
        print(f"\nReady to process {len(available_files)} files") 
        batch_process_files(base_dir)
    else:
        print("No available files found to process")