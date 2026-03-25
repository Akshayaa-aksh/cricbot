import json
import os
import time
import certifi

os.environ["PINECONE_DISABLE_SSL"] = "false"

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ======= Real Embeddings (same model as search.py) =======
print("⏳ Loading real HuggingFace embeddings...")
try:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'local_files_only': True})
except Exception:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("✅ Embeddings loaded!")

# ======= Init Pinecone =======
print("⏳ Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "cricket-bot"
DIMENSION  = 384  # all-MiniLM-L6-v2 always outputs 384 dimensions

# Delete old index if it exists (old one had wrong dimension)
existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME in existing:
    print(f"🗑️  Deleting old index '{INDEX_NAME}' (wrong dimension)...")
    pc.delete_index(INDEX_NAME)
    time.sleep(10)  # wait for deletion to complete

print(f"🔧 Creating new index '{INDEX_NAME}' with dimension {DIMENSION}...")
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Wait for index to be ready
print("⏳ Waiting for index to be ready...")
while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(2)

index = pc.Index(INDEX_NAME)
print("✅ Pinecone index ready!")

# ======= Helper: embed a piece of text =======
def embed_text(text):
    return embedder.embed_query(text)

# ======= Helper: upsert in batches (Pinecone limit = 100 per batch) =======
def upsert_batch(vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"   ✅ Uploaded {min(i+batch_size, len(vectors))}/{len(vectors)}")

# ======= 1. Players =======
print("\n📤 Uploading players...")
with open("players.json", "r", encoding="utf-8") as f:
    players = json.load(f)["teams"]

player_vectors = []
for p in players:
    text = f"{p['name']} is a cricket player. Slug: {p.get('slug', '')}."
    player_vectors.append({
        "id": f"player_{p['id']}",
        "values": embed_text(text),
        "metadata": {
            "type":   "player",
            "name":   p["name"],
            "slug":   p.get("slug", ""),
            "image":  p.get("image", ""),
            "text":   text
        }
    })

upsert_batch(player_vectors)
print(f"✅ {len(player_vectors)} players uploaded!")

# ======= Helper: load matches from a JSON file =======
def load_and_upload_matches(filename, match_type):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  {filename} not found, skipping.")
        return

    schedules = data.get("response", {}).get("schedules", [])
    vectors   = []
    counter   = 0

    for s in schedules:
        wrapper = s.get("scheduleAdWrapper", {})
        for m in wrapper.get("matchScheduleList", []):
            series = m.get("seriesName", "")
            for info in m.get("matchInfo", []):
                team1  = info.get("team1", {}).get("teamName", "")
                team2  = info.get("team2", {}).get("teamName", "")
                venue  = info.get("venueInfo", {}).get("city", "")
                desc   = info.get("matchDesc", "")
                date   = info.get("startDate", "")

                text = f"{team1} vs {team2} {desc} in {series} at {venue}."

                vectors.append({
                    "id": f"{match_type}_{counter}",
                    "values": embed_text(text),
                    "metadata": {
                        "type":   match_type,
                        "team1":  team1,
                        "team2":  team2,
                        "venue":  venue,
                        "series": series,
                        "desc":   desc,
                        "date":   date,
                        "text":   text
                    }
                })
                counter += 1

    print(f"\n📤 Uploading {len(vectors)} {match_type} matches...")
    upsert_batch(vectors)
    print(f"✅ {len(vectors)} {match_type} matches uploaded!")

# ======= 2. International matches =======
load_and_upload_matches("international.json", "international")

# ======= 3. League/IPL matches =======
load_and_upload_matches("league.json", "league")

# ======= 4. Women's matches =======
load_and_upload_matches("women.json", "women")

# ======= Done =======
stats = index.describe_index_stats()
print(f"\n🎉 All done! Total vectors in Pinecone: {stats['total_vector_count']}")
print("Now run search.py to query it!")