import requests
from bs4 import BeautifulSoup
import os

# Create folders
os.makedirs("docs/medlineplus", exist_ok=True)

# MedlinePlus pages to scrape
pages = {
    "menstruation.txt":  "https://medlineplus.gov/menstruation.html",
    "dysmenorrhea.txt":  "https://medlineplus.gov/periodpain.html",
    "amenorrhea.txt":    "https://medlineplus.gov/amenorrhea.html",
    "pcos.txt":          "https://medlineplus.gov/polycysticovarysyndrome.html",
}

for filename, url in pages.items():
    print(f"Downloading: {url}")
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Extract main content only
    content = soup.find('div', {'id': 'topic-summary'})
    if content:
        text = content.get_text(separator='\n', strip=True)
    else:
        text = soup.get_text(separator='\n', strip=True)

    filepath = f"docs/medlineplus/{filename}"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved: {filepath}")

print("\nAll done!")