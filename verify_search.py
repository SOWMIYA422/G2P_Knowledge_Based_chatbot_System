import requests
res = requests.get('http://localhost:8000/api/search?query=high protein')
data = res.json()
if data:
    print(f"First: {data[0].get('Variety', data[0].get('Genotype'))}")
    print(f"Grain weight: {data[0].get('Grain_weight')}")
else:
    print("No results")
