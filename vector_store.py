import json
import os
import certifi

# Fix for Windows SSL_CERT_FILE invalid argument error
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'local_files_only': True})
except Exception:
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = []

# ✅ 1. PLAYERS.JSON
with open("players.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    players = data.get("teams", [])

    for p in players:
        text = f"{p.get('name')} is a cricket player. slug is {p.get('slug')}"
        documents.append(Document(page_content=text, metadata={"type": "player"}))


# ✅ 2. WOMEN.JSON
with open("women.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    schedules = data.get("response", {}).get("schedules", [])

    for s in schedules:
        wrapper = s.get("scheduleAdWrapper", {})
        matches = wrapper.get("matchScheduleList", [])

        for m in matches:
            series = m.get("seriesName", "")
            for info in m.get("matchInfo", []):
                team1 = info.get("team1", {}).get("teamName", "")
                team2 = info.get("team2", {}).get("teamName", "")
                venue = info.get("venueInfo", {}).get("city", "")
                desc = info.get("matchDesc", "")

                text = f"{team1} vs {team2} {desc} in {series} at {venue}"
                documents.append(Document(page_content=text, metadata={"type": "women_match"}))


# ✅ 3. INTERNATIONAL.JSON
with open("international.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    schedules = data.get("response", {}).get("schedules", [])

    for s in schedules:
        wrapper = s.get("scheduleAdWrapper", {})
        matches = wrapper.get("matchScheduleList", [])

        for m in matches:
            series = m.get("seriesName", "")
            for info in m.get("matchInfo", []):
                team1 = info.get("team1", {}).get("teamName", "")
                team2 = info.get("team2", {}).get("teamName", "")
                venue = info.get("venueInfo", {}).get("city", "")
                desc = info.get("matchDesc", "")

                text = f"{team1} vs {team2} {desc} in {series} at {venue}"
                documents.append(Document(page_content=text, metadata={"type": "international_match"}))


# ✅ 4. LEAGUE.JSON
with open("league.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    schedules = data.get("response", {}).get("schedules", [])

    for s in schedules:
        wrapper = s.get("scheduleAdWrapper", {})
        matches = wrapper.get("matchScheduleList", [])

        for m in matches:
            series = m.get("seriesName", "")
            for info in m.get("matchInfo", []):
                team1 = info.get("team1", {}).get("teamName", "")
                team2 = info.get("team2", {}).get("teamName", "")
                venue = info.get("venueInfo", {}).get("city", "")
                desc = info.get("matchDesc", "")

                text = f"{team1} vs {team2} {desc} in {series} at {venue}"
                documents.append(Document(page_content=text, metadata={"type": "league_match"}))


print(f"Total documents: {len(documents)}")

# ✅ Create DB
db = Chroma.from_documents(
    documents,
    embedding,
    persist_directory="db"
)

db.persist()

print("🔥 Vector DB ready ")