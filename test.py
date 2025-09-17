
import requests

texts = [
    "A powerful hurricane is moving toward the Florida coastline, expected landfall within 12 hours.",
    "Heavy rainfall has caused flash floods in downtown streets, blocking traffic and damaging houses.",
    "The city is hosting a cultural festival this weekend with music and food stalls.",
    "Cyclone Idai has intensified into a Category 4 storm, urgent evacuation orders issued for coastal residents.",
    "River levels are rising steadily, officials warn of possible flooding in low-lying areas in the next few days.",
    "Weather department reports cloudy skies with mild rainfall, no immediate danger expected."
]

for text in texts:
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": text})
    print(f"Input: {text}")
    print("Prediction:", response.json())
    print("-" * 80)
