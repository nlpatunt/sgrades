import pandas as pd
import requests
from bs4 import BeautifulSoup

HF_TOKEN = "REMOVED_KEY"

# Test with the first URL
test_url = 'https://huggingface.co/datasets/nlpatunt/dataset_rubrics/raw/main/ASAP_SAS/EssaySet6/essay_set_6_prompt.html'

headers = {
    'Authorization': f'Bearer {HF_TOKEN}'
}

response = requests.get(test_url, headers=headers)
print(f"Status: {response.status_code}")
print(f"\nFirst 500 chars of response:")
print(response.text[:500])

soup = BeautifulSoup(response.text, 'html.parser')

print(f"\n\nAll h2 tags:")
for h2 in soup.find_all('h2'):
    print(f"  '{h2.get_text()}'")

print(f"\n\nAll p tags (first 5):")
for i, p in enumerate(soup.find_all('p')[:5]):
    print(f"  {i+1}. '{p.get_text()[:100]}'")

print(f"\n\nAll strong tags (first 5):")
for i, strong in enumerate(soup.find_all('strong')[:5]):
    print(f"  {i+1}. '{strong.get_text()[:100]}'")