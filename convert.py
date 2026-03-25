import json

# Load JSON
with open("players.json", "r", encoding="utf-8") as f:
    data = json.load(f)

players = data["teams"]

documents = []

# Convert to text
for player in players:
    text = f"{player['name']} is an Indian cricket player."
    documents.append(text)

# Print all
for doc in documents:
    print(doc)

# Count
print("\nTotal players:", len(documents))