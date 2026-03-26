import json
import re
import warnings
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder          # ✅ NEW: for re-ranking
from dotenv import load_dotenv

from live_cricket import (
    get_live_scores,
    get_upcoming_matches,
    get_recent_results,
    get_cricket_news,
    search_player_live,
)

load_dotenv()
warnings.filterwarnings("ignore")

# ======= Embeddings =======
print("⏳ Loading Embeddings...")
try:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'local_files_only': True})
except Exception:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ======= Cross-Encoder Re-Ranker =======
# This model scores (query, passage) pairs directly — much more accurate than
# bi-encoder cosine similarity alone.
print("⏳ Loading Cross-Encoder Re-Ranker...")
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    print("✅ Re-Ranker loaded!")
except Exception as e:
    reranker = None
    print(f"⚠️  Re-Ranker failed to load (will skip re-ranking): {e}")

# ======= Pinecone Setup =======
print("⏳ Connecting to Pinecone...")
pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("cricket-bot")
print("✅ Pinecone connected!")

# ======= Groq LLM Setup =======
print("⏳ Initializing Groq LLM...")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

prompt_template = PromptTemplate.from_template(
    """You are CricBot, an enthusiastic and deeply knowledgeable cricket expert chatbot.
You love cricket passionately and always give detailed, engaging, and accurate answers.

Use the context below (retrieved from live API + our database) to answer the question.
If the context includes live scores, schedules, or news — summarise and present it clearly.
If the context doesn't have enough info, use your own cricket knowledge freely.
Always be friendly, use cricket terms naturally, and keep answers concise but rich.

Context:
{context}

User's question: {question}

CricBot's answer:"""
)

# ======= Load players JSON =======
with open("players.json", "r", encoding="utf-8") as f:
    players_list = json.load(f)["teams"]

# ======= Load match schedules from local JSON files =======
def load_matches(filename, match_type):
    matches = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("response", {}).get("schedules", []):
            wrapper = s.get("scheduleAdWrapper", {})
            for m in wrapper.get("matchScheduleList", []):
                series = m.get("seriesName", "")
                for info in m.get("matchInfo", []):
                    matches.append({
                        "type":   match_type,
                        "team1":  info.get("team1", {}).get("teamName", ""),
                        "team2":  info.get("team2", {}).get("teamName", ""),
                        "venue":  info.get("venueInfo", {}).get("city", ""),
                        "date":   info.get("startDate", ""),
                        "desc":   info.get("matchDesc", ""),
                        "series": series
                    })
    except FileNotFoundError:
        pass
    return matches

all_matches = (
    load_matches("international.json", "International") +
    load_matches("league.json",        "League/IPL")    +
    load_matches("women.json",         "Women's")
)

# ======= LIVE INTENT KEYWORDS =======
LIVE_SCORE_KEYWORDS = [
    "live", "score", "playing now", "current score", "today score",
    "match today", "who is batting", "who is bowling", "wicket",
    "run rate", "live match", "scorecard", "over by over"
]

UPCOMING_KEYWORDS = [
    "upcoming", "next match", "schedule", "fixture", "when does",
    "when is", "when will", "next game", "india next",
    "ipl 2026", "ipl start", "ipl schedule", "ipl date",
    "ipl 2025", "starting", "begin", "start date"
]

RECENT_RESULT_KEYWORDS = [
    "result", "who won", "last match", "yesterday", "recent match",
    "latest result", "final score", "match result", "who beat",
    "winner", "won the match"
]

NEWS_KEYWORDS = [
    "news", "latest news", "update", "cricket news", "what happened",
    "recent news", "today news", "headlines", "latest update"
]


def detect_live_intent(query):
    """Returns 'live_score', 'upcoming', 'recent', 'news', or None."""
    q = query.lower()
    if any(k in q for k in LIVE_SCORE_KEYWORDS):
        return "live_score"
    if any(k in q for k in UPCOMING_KEYWORDS):
        return "upcoming"
    if any(k in q for k in RECENT_RESULT_KEYWORDS):
        return "recent"
    if any(k in q for k in NEWS_KEYWORDS):
        return "news"
    return None


def fetch_live_context(intent):
    """Fetch real-time data from Cricbuzz based on detected intent."""
    print(f"🌐 Fetching live data ({intent})...")
    if intent == "live_score":
        return get_live_scores()
    elif intent == "upcoming":
        return get_upcoming_matches()
    elif intent == "recent":
        return get_recent_results()
    elif intent == "news":
        return get_cricket_news()
    return ""


# ======= Helpers =======
def clean_query(q):
    q = q.lower().strip()
    q = re.sub(r"(who is|tell me about|info about|give details of|what is|details of|show me)", "", q)
    return q.replace("?", "").strip()


def embed_text(text):
    return embedder.embed_query(text)


def pinecone_search(query, top_k=10):
    """
    Fetch more candidates from Pinecone (top_k=10 by default) so the
    re-ranker has a larger pool to pick the best ones from.
    """
    vector  = embed_text(query)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results.get("matches", [])


# ======= ✅ RE-RANKING FUNCTION =======
def rerank_results(query, matches, top_n=5, min_score_threshold=0.0):
    """
    Re-rank Pinecone matches using a Cross-Encoder model.

    How it works:
      - Pinecone returns results ranked by bi-encoder cosine similarity.
        Bi-encoders are fast but less accurate because query and passage
        are encoded independently.
      - A Cross-Encoder reads the (query + passage) TOGETHER, giving a
        much more precise relevance score.
      - We score every candidate, sort by cross-encoder score, and keep
        only the top_n most relevant ones above the threshold.

    Args:
        query           : The user's original question.
        matches         : Raw list of Pinecone match dicts.
        top_n           : How many results to keep after re-ranking.
        min_score_threshold : Drop results below this cross-encoder score.

    Returns:
        List of dicts: [{"text": ..., "pinecone_score": ..., "rerank_score": ...}, ...]
    """
    if not reranker:
        # Fallback: just use Pinecone scores if re-ranker isn't available
        fallback = []
        for r in matches:
            text = r.get("metadata", {}).get("text", "")
            if text and r.get("score", 0) > 0.3:
                fallback.append({
                    "text":           text,
                    "pinecone_score": r.get("score", 0),
                    "rerank_score":   r.get("score", 0),   # same as pinecone score
                })
        return fallback[:top_n]

    # Build (query, passage) pairs for the cross-encoder
    candidates = []
    for r in matches:
        text = r.get("metadata", {}).get("text", "")
        if text:
            candidates.append({
                "text":           text,
                "pinecone_score": r.get("score", 0),
            })

    if not candidates:
        return []

    # Score all candidates with the cross-encoder in one batch (efficient)
    pairs         = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)          # returns a list of floats

    # Attach rerank scores to each candidate
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)

    # Sort by cross-encoder score (highest = most relevant)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Filter by threshold and keep top_n
    filtered = [c for c in candidates if c["rerank_score"] >= min_score_threshold]
    return filtered[:top_n]


def get_player_by_name(name):
    name = name.lower().strip()
    for p in players_list:
        if p["name"].lower() == name:
            return p
    for p in players_list:
        if name in p["name"].lower().split():
            return p
    for p in players_list:
        if name in p["name"].lower():
            return p
    return None


def display_player_info(player):
    print("\n" + "=" * 60)
    print("✅ PLAYER FOUND")
    print("=" * 60)
    print(f"📌 Name      : {player.get('name', 'N/A')}")
    print(f"🆔 Player ID : {player.get('id', 'N/A')}")
    print(f"🏷️  Slug      : {player.get('slug', 'N/A')}")
    print(f"📷 Photo     : {player.get('image', 'N/A')}")
    print("=" * 60 + "\n")


def find_matches_for_query(query):
    query = query.lower()
    found = []
    for m in all_matches:
        searchable = f"{m['team1']} {m['team2']} {m['series']} {m['venue']} {m['type']}".lower()
        if any(word in searchable for word in query.split() if len(word) > 2):
            found.append(m)
    return found[:5]


def display_matches(matches):
    print(f"\n{'='*60}")
    print("🏏 Related Matches from Local DB")
    print(f"{'='*60}")
    for m in matches:
        print(f"  [{m['type']}] {m['team1']} vs {m['team2']}")
        print(f"           {m['desc']} | {m['series']}")
        print(f"           📍 {m['venue']}")
        print()


# STRICT predefined answers — exact match only
def get_predefined_answer(query):
    answers = {
        "what is cricket":       "Cricket is a bat-and-ball sport played between two teams of 11 players. It's the second most popular sport in the world!",
        "what is ipl":           "The IPL (Indian Premier League) is the world's biggest T20 cricket league with 10 franchise teams, played every year in India.",
        "father of cricket":     "WG Grace is traditionally called the Father of Cricket. In India, Sachin Tendulkar is lovingly called the God of Cricket.",
        "god of cricket":        "Sachin Tendulkar is the undisputed God of Cricket! 100 international centuries, 34,357 runs — the GOAT!",
        "best batsman":          "Virat Kohli is widely regarded as the best batsman of his generation with 80+ international centuries.",
        "best bowler":           "Jasprit Bumrah is the No.1 ranked bowler in the world, known for his unique action and deadly yorkers.",
        "how many players":      "Each cricket team has 11 players on the field. Squads are typically 15-16 players from which the playing XI is chosen.",
        "how many overs in t20": "A T20 match has 20 overs per innings — each team bats once for a maximum of 20 overs.",
        "how many overs in odi": "An ODI (One Day International) has 50 overs per innings for each team.",
    }
    for key, value in answers.items():
        if query.strip() == key:
            return value
    return None


# ======= Main Chat Loop =======
print("\n" + "🏏" * 30)
print("   🤖 CricBot is Ready! Ask me anything about cricket!")
print("   (Powered by Cricbuzz Live + Pinecone + Cross-Encoder Re-Ranking + Groq)")
print("🏏" * 30)
print("   Type 'exit' or 'quit' to leave\n")

while True:
    try:
        user_input = input("You: ").strip()
    except EOFError:
        break

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n🤖 CricBot: Goodbye! May your team always win! 🏆\n")
        break

    cleaned = clean_query(user_input)
    context_parts = []

    # ── 1. Live intent detection → Cricbuzz API (CHECKED FIRST) ─────────
    live_intent = detect_live_intent(user_input)
    if live_intent:
        live_data = fetch_live_context(live_intent)
        if live_data:
            print(f"\n{live_data}\n")
            context_parts.append(f"Live data from Cricbuzz API:\n{live_data}")

    # ── 2. Predefined answers (only if NO live intent detected) ──────────
    if not live_intent:
        predefined = get_predefined_answer(cleaned)
        if predefined:
            print(f"\n🤖 CricBot: {predefined}\n")
            continue

    # ── 3. Local player name lookup ──────────────────────────────────────
    player = get_player_by_name(cleaned)
    if player:
        display_player_info(player)
        live_player = search_player_live(player["name"])
        player_ctx  = f"Player: {player.get('name')}, ID: {player.get('id')}, Slug: {player.get('slug')}"
        if live_player:
            player_ctx += f"\nLive profile: {live_player['summary']}"
        context_parts.append(f"Player info:\n{player_ctx}")

    # ── 4. Local match schedule lookup ───────────────────────────────────
    matches = find_matches_for_query(cleaned)
    if matches:
        display_matches(matches)
        match_ctx = "\n".join(
            f"{m['team1']} vs {m['team2']} ({m['desc']}) in {m['series']} at {m['venue']}"
            for m in matches
        )
        context_parts.append(f"Match schedules (local DB):\n{match_ctx}")

    # ── 5. Pinecone search → Cross-Encoder Re-Ranking ────────────────────
    print("🔍 Searching Pinecone (fetching 10 candidates for re-ranking)...")
    raw_matches    = pinecone_search(user_input, top_k=10)   # fetch more candidates
    reranked       = rerank_results(user_input, raw_matches, top_n=5, min_score_threshold=0.0)

    if reranked:
        print(f"✅ Re-ranked to top {len(reranked)} results")
        print(f"\n{'─'*60}")
        print("📊 Re-Ranked Results (Cross-Encoder Scores):")
        pine_parts = []
        for i, r in enumerate(reranked, 1):
            print(f"  {i}. [Pinecone: {r['pinecone_score']:.3f} → Rerank: {r['rerank_score']:.3f}] "
                  f"{r['text'][:80]}...")
            pine_parts.append(
                f"- {r['text']} "
                f"(pinecone_score: {r['pinecone_score']:.2f}, rerank_score: {r['rerank_score']:.2f})"
            )
        print(f"{'─'*60}\n")
        context_parts.append("Re-ranked Pinecone results:\n" + "\n".join(pine_parts))
    else:
        print("⚠️  No relevant results found in Pinecone after re-ranking.")

    # ── 6. Call Groq LLM with all combined context ───────────────────────
    full_context = "\n\n".join(context_parts) if context_parts else "No specific data found."

    print("🤖 CricBot: ", end="", flush=True)
    try:
        prompt   = prompt_template.format(context=full_context, question=user_input)
        response = llm.invoke(prompt)
        print(response.content)
    except Exception as e:
        print(f"\n❌ Groq Error: {e}")
        print("(Check your GROQ_API_KEY in .env)")

    print()