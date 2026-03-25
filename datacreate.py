import pandas as pd
import numpy as np
import random
from faker import Faker
import csv

# Initialize Faker for generating realistic data
fake = Faker()


def generate_indian_rice_dataset(
    num_varieties=500, output_file="indian_rice_varieties.csv"
):
    """
    Generate a comprehensive dataset of Indian rice varieties
    """

    # Real Indian rice varieties (expanded list)
    base_varieties = [
        "Pusa Basmati 1121",
        "Pusa Basmati 1509",
        "Pusa Basmati 6",
        "Pusa Basmati 1",
        "Sona Masuri",
        "Swarna",
        "MTU 1010",
        "IR 64",
        "Jaya",
        "Ratna",
        "Ponni",
        "Ambemohar",
        "Govind Bhog",
        "Kalanamak",
        "Jeeragasamba",
        "Lal Matta",
        "Navara",
        "Basmati 370",
        "Dubraj",
        "Kolam",
        "Kala Namak",
        "Gobindobhog",
        "Chinnor",
        "Indrayani",
        "Karjat 3",
        "Kanak Jeera",
        "Kataribhog",
        "Badshah Bhog",
        "Hansraj",
        "Sharbati",
        "Sugandha",
        "Krishna Hamsa",
        "Luchai",
        "Moti",
        "Neela",
        "Rajendra Bhagwati",
        "Rajendra Sweta",
        "Rajendra Kasturi",
        "Sahyadri",
        "Pusa Sugandha 5",
        "Pusa RH 10",
        "DRR 44",
        "Ajay",
        "Amrit",
        "Annada",
        "Arize",
        "Akshay",
        "BPT 5204",
        "BPT 2231",
        "BPT 3291",
        "Co 51",
        "Co 52",
        "DRRH 3",
        "HKR 47",
        "JGL 1798",
        "KPH 457",
        "NLR 34449",
        "PR 113",
        "Pusa 44",
        "Pusa 1121",
        "Pusa 1509",
        "Rajendra",
        "Samba Mahsuri",
        "Satya",
        "Shaktiman",
        "Tellahamsa",
        "Vandana",
        "VL Dhan 85",
        "VL Dhan 86",
        "VL Dhan 87",
        "Zinco Rice 1",
        "Naveen",
        "Rasi",
        "MTU 1001",
        "MTU 7029",
        "MTU 1010",
        "DRR Dhan 48",
        "DRR Dhan 49",
        "RDN 01-2",
        "RDN 01-3",
        "Krishna",
        "Ganga",
        "Yamuna",
        "Brahmaputra",
        "Godavari",
        "Kaveri",
        "Narmada",
        "Tapti",
        "Mahanadi",
        "Sutlej",
        "Chenab",
        "Ravi",
        "Beas",
        "Jhelum",
        "Indus",
        "Aravalli",
        "Vindhya",
        "Satpura",
        "Western Ghats",
        "Eastern Ghats",
        "Deccan",
        "Malabar",
        "Coromandel",
        "Konkan",
        "Doab",
    ]

    # Additional generated variety names
    additional_varieties = []
    for i in range(num_varieties - len(base_varieties)):
        prefix = random.choice(
            ["Pusa", "MTU", "DRR", "VL", "Rajendra", "NLR", "Co", "BPT"]
        )
        number = random.randint(1, 9999)
        suffix = random.choice(
            ["", "Gold", "Premium", "Select", "Elite", "Plus", "Super"]
        )
        additional_varieties.append(
            f"{prefix} {number}{' ' + suffix if suffix else ''}"
        )

    all_varieties = (
        base_varieties + additional_varieties[: num_varieties - len(base_varieties)]
    )

    # Indian states for geographical distribution
    indian_states = [
        "Punjab",
        "Haryana",
        "Uttar Pradesh",
        "West Bengal",
        "Andhra Pradesh",
        "Telangana",
        "Tamil Nadu",
        "Karnataka",
        "Kerala",
        "Maharashtra",
        "Gujarat",
        "Rajasthan",
        "Madhya Pradesh",
        "Bihar",
        "Odisha",
        "Assam",
        "Chhattisgarh",
        "Jharkhand",
        "Uttarakhand",
        "Himachal Pradesh",
    ]

    # Soil types in India
    soil_types = [
        "Alluvial",
        "Clay",
        "Loam",
        "Sandy Loam",
        "Red",
        "Black",
        "Laterite",
        "Mountain",
    ]

    # Groups/Classification
    groups = ["Indica", "Japonica", "Aromatic", "Hybrid", "Traditional", "Improved"]

    # Generate data
    data = []

    for i, variety in enumerate(all_varieties[:num_varieties]):
        # Generate unique SampleID
        sample_id = f"IND{i + 1:04d}"

        # Variety info
        country = "India"
        group = random.choice(groups)

        # Generate SNP data (20 SNPs with values 0, 1, or 2)
        snp_data = {}
        for j in range(1, 21):
            snp_name = f"OsSNP{j:03d}"
            # Create realistic SNP patterns (some correlation with traits)
            if j in [1, 5, 8, 12]:  # Drought tolerance related SNPs
                base_value = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
            elif j in [2, 6, 10, 14]:  # Yield related SNPs
                base_value = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
            elif j in [3, 7, 11, 15]:  # Height related SNPs
                base_value = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
            elif j in [4, 9, 13, 16]:  # Grain weight related SNPs
                base_value = random.choices([0, 1, 2], weights=[0.3, 0.3, 0.4])[0]
            else:
                base_value = random.choices([0, 1, 2], weights=[0.33, 0.34, 0.33])[0]

            # Add some noise/variation
            final_value = max(0, min(2, base_value + random.choice([-1, 0, 1])))
            snp_data[snp_name] = final_value

        # Coverage (sequencing coverage)
        coverage = round(random.uniform(0.5, 4.0), 2)

        # Heterozygosity
        heterozygosity = round(random.uniform(0.15, 0.35), 3)

        # Phenotypic traits with correlations to SNPs
        # Yield_per_plant (g/plant) - higher with certain SNPs
        yield_base = 25.0
        yield_boost = (
            sum([snp_data.get(f"OsSNP{j:03d}", 0) for j in [2, 6, 10, 14]]) * 1.5
        )
        yield_per_plant = round(yield_base + yield_boost + random.uniform(-5, 10), 2)
        yield_per_plant = max(
            15, min(65, yield_per_plant)
        )  # Bound between 15-65 g/plant

        # Height (cm) - taller with certain SNPs
        height_base = 90.0
        height_boost = (
            sum([snp_data.get(f"OsSNP{j:03d}", 0) for j in [3, 7, 11, 15]]) * 2.0
        )
        height = round(height_base + height_boost + random.uniform(-20, 30), 1)
        height = max(50, min(150, height))  # Bound between 50-150 cm

        # Grain_weight (g/1000 grains) - heavier with certain SNPs
        grain_base = 25.0
        grain_boost = (
            sum([snp_data.get(f"OsSNP{j:03d}", 0) for j in [4, 9, 13, 16]]) * 1.0
        )
        grain_weight = round(grain_base + grain_boost + random.uniform(-5, 8), 2)
        grain_weight = max(18, min(40, grain_weight))  # Bound between 18-40 g

        # Drought_Tolerance (0 or 1) - influenced by certain SNPs
        drought_score = sum([snp_data.get(f"OsSNP{j:03d}", 0) for j in [1, 5, 8, 12]])
        drought_tolerance = 1 if drought_score >= 5 else 0

        # Environmental data based on Indian regions
        state = random.choice(indian_states)

        # Rainfall based on region
        if state in ["Kerala", "West Bengal", "Assam", "Odisha"]:
            rainfall = round(random.uniform(1200, 3000), 1)  # High rainfall regions
        elif state in ["Rajasthan", "Gujarat", "Haryana", "Punjab"]:
            rainfall = round(random.uniform(300, 800), 1)  # Low rainfall regions
        else:
            rainfall = round(random.uniform(800, 1500), 1)  # Medium rainfall

        # Temperature based on region
        if state in ["Himachal Pradesh", "Uttarakhand"]:
            temperature = round(random.uniform(15, 25), 1)  # Cool regions
        elif state in ["Rajasthan", "Gujarat", "Maharashtra"]:
            temperature = round(random.uniform(25, 35), 1)  # Warm regions
        else:
            temperature = round(random.uniform(20, 30), 1)  # Moderate regions

        # Soil type based on region
        if state in ["Punjab", "Haryana", "Uttar Pradesh", "West Bengal"]:
            soil = "Alluvial"
        elif state in ["Madhya Pradesh", "Maharashtra", "Gujarat"]:
            soil = random.choice(["Black", "Clay"])
        elif state in ["Karnataka", "Tamil Nadu", "Kerala"]:
            soil = random.choice(["Red", "Laterite"])
        else:
            soil = random.choice(soil_types)

        # Create record
        record = {
            "SampleID": sample_id,
            "Variety": variety,
            "Country": country,
            "Group": group,
            **snp_data,
            "Coverage": coverage,
            "Heterozygosity": heterozygosity,
            "Yield_per_plant": yield_per_plant,
            "Height": height,
            "Grain_weight": grain_weight,
            "Drought_Tolerance": drought_tolerance,
            "Rainfall_mm": rainfall,
            "Temperature_C": temperature,
            "Soil_Type": soil,
            "State": state,  # Adding state for better geographical context
        }

        data.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False)

    # Also create an Excel version
    excel_file = output_file.replace(".csv", ".xlsx")
    df.to_excel(excel_file, index=False)

    return df, output_file, excel_file


def generate_statistics(df):
    """Generate summary statistics for the dataset"""
    stats = {
        "Total Varieties": df["Variety"].nunique(),
        "Total Samples": len(df),
        "Average Yield (g/plant)": df["Yield_per_plant"].mean(),
        "Average Height (cm)": df["Height"].mean(),
        "Average Grain Weight (g)": df["Grain_weight"].mean(),
        "Drought Tolerant Varieties": df["Drought_Tolerance"].sum(),
        "States Represented": df["State"].nunique(),
        "Soil Types": df["Soil_Type"].nunique(),
    }

    return stats


def main():
    """Main function to generate and display dataset"""
    print("🌾 Generating Indian Rice Varieties Dataset...")

    # Generate 500 varieties dataset
    df, csv_file, excel_file = generate_indian_rice_dataset(num_varieties=500)

    # Generate statistics
    stats = generate_statistics(df)

    print(f"\n✅ Dataset Generated Successfully!")
    print(f"📊 Dataset Statistics:")
    for key, value in stats.items():
        print(
            f"   {key}: {value:.2f}"
            if isinstance(value, float)
            else f"   {key}: {value}"
        )

    print(f"\n📁 Files Created:")
    print(f"   1. {csv_file} (CSV format)")
    print(f"   2. {excel_file} (Excel format)")

    # Display sample of the data
    print(f"\n📋 Sample Data (first 5 varieties):")
    print(
        df[
            [
                "SampleID",
                "Variety",
                "State",
                "Yield_per_plant",
                "Height",
                "Drought_Tolerance",
            ]
        ].head()
    )

    # Create a summary report
    print("\n🌍 Geographical Distribution:")
    state_dist = df["State"].value_counts().head(10)
    for state, count in state_dist.items():
        print(f"   {state}: {count} varieties")

    print("\n🌱 Trait Distribution:")
    print(f"   High Yield (>40g/plant): {(df['Yield_per_plant'] > 40).sum()} varieties")
    print(f"   Tall Varieties (>120cm): {(df['Height'] > 120).sum()} varieties")
    print(f"   Heavy Grains (>35g): {(df['Grain_weight'] > 35).sum()} varieties")

    return df


if __name__ == "__main__":
    df = main()

    # Instructions for downloading
    print("\n📥 To download the dataset directly:")
    print("1. Run this script to generate the files")
    print("2. Files will be saved in the current directory")
    print("3. You can load them directly into your project")
