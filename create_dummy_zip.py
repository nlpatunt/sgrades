# create_dummy_zip.py - Run this to create a test ZIP file

import zipfile
import csv
import io
import random

def create_dummy_csv_content(dataset_name, num_essays=50):
    """Create dummy CSV content for a dataset"""
    
    # Generate dummy essay IDs and scores
    essays = []
    for i in range(num_essays):
        essay_id = f"{dataset_name}_{i:03d}"
        # Random score between 1-6 with some variation
        predicted_score = round(random.uniform(1.0, 6.0), 2)
        essays.append({
            'essay_id': essay_id,
            'predicted_score': predicted_score
        })
    
    # Convert to CSV string
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['essay_id', 'predicted_score'])
    writer.writeheader()
    writer.writerows(essays)
    
    return output.getvalue()

def create_dummy_benchmark_zip():
    """Create a dummy ZIP file with all 12 datasets"""
    
    # Your 12 datasets
    datasets = [
        'ASAP-AES',
        'ASAP-SAS', 
        'rice_chem',
        'CSEE',
        'EFL',
        'grade_like_a_human_dataset_os',
        'persuade_2',
        'ASAP2',
        'ASAP_plus_plus',
        'SciEntSBank',
        'BEEtlE',
        'automatic_short_answer_grading'
    ]
    
    zip_filename = 'dummy_benchmark_results.zip'
    
    print(f"🔄 Creating dummy ZIP file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for dataset_name in datasets:
            # Create CSV content
            csv_content = create_dummy_csv_content(dataset_name, num_essays=random.randint(30, 100))
            
            # Add to ZIP
            csv_filename = f"{dataset_name}.csv"
            zipf.writestr(csv_filename, csv_content)
            
            print(f"   ✅ Added {csv_filename} with {len(csv_content.splitlines())-1} essays")
    
    print(f"🎉 Created {zip_filename} successfully!")
    print(f"📁 ZIP contains {len(datasets)} CSV files")
    print(f"📤 You can now upload this ZIP to test your benchmark submission")
    
    return zip_filename

def create_single_test_csv():
    """Create a single CSV for testing single submissions"""
    
    csv_content = create_dummy_csv_content('ASAP-AES', num_essays=20)
    
    filename = 'test_ASAP-AES.csv'
    with open(filename, 'w') as f:
        f.write(csv_content)
    
    print(f"📄 Created {filename} for single dataset testing")
    return filename

if __name__ == "__main__":
    print("🚀 Creating dummy test files for BESESR benchmark...")
    
    # Create ZIP file for complete benchmark
    zip_file = create_dummy_benchmark_zip()
    
    print()
    
    # Create single CSV for testing
    single_file = create_single_test_csv()
    
    print()
    print("📋 Files created:")
    print(f"   1. {zip_file} - Use for complete benchmark submission")
    print(f"   2. {single_file} - Use for single dataset testing")
    print()
    print("🧪 Test Instructions:")
    print("   1. Upload the ZIP file using the main benchmark form")
    print("   2. Upload the single CSV using the testing form")
    print("   3. The ZIP should appear on leaderboard, single CSV should not")