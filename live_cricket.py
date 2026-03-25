import os
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY  = os.getenv("a98bd70c92msh4d129d01f34be07p18f8e6jsnbce02380282e")
RAPIDAPI_HOST = "cricbuzz-cricket.p.rapidapi.com"

HEADERS = {
    "X-RapidAPI-Key":  RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

BASE_URL = "https://cricbuzz-cricket.p.rapidapi.com"


# ======= Helper =======
def _get(endpoint, params=None):
    """Make a GET request to Cricbuzz API. Returns dict or None on failure."""
    try:
        url      = f"{BASE_URL}/{endpoint}"
        response = requests.get(url, headers=HEADERS, params=params, timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("⚠️  Cricbuzz API timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"⚠️  Cricbuzz API error: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Unexpected error calling Cricbuzz: {e}")
        return None


# ======= 1. Live / recent scores =======
def get_live_scores():
    """
    Returns a formatted string of live and recent match scores.
    Covers international + domestic + T20 leagues.
    """
    data = _get("matches/v1/live")
    if not data:
        return "Sorry, I couldn't fetch live scores right now. Please try again in a moment!"

    matches = []
    type_map = data.get("typeMatches", [])

    for t in type_map:
        match_type = t.get("matchType", "")
        for series in t.get("seriesMatches", []):
            series_info = series.get("seriesAdWrapper") or series.get("adDetail")
            if not series_info:
                continue
            series_name = series_info.get("seriesName", "")
            for m in series_info.get("matches", []):
                info   = m.get("matchInfo", {})
                score  = m.get("matchScore", {})

                team1  = info.get("team1", {}).get("teamSName", "?")
                team2  = info.get("team2", {}).get("teamSName", "?")
                status = info.get("status", "")
                state  = info.get("state",  "")
                venue  = info.get("venueInfo", {}).get("city", "")

                # innings scores
                t1_score = score.get("team1Score", {})
                t2_score = score.get("team2Score", {})

                def fmt_innings(s):
                    inn = s.get("inngs1", {})
                    if not inn:
                        return ""
                    r, w, o = inn.get("runs","?"), inn.get("wickets","?"), inn.get("overs","?")
                    return f"{r}/{w} ({o} ov)"

                t1_str = fmt_innings(t1_score)
                t2_str = fmt_innings(t2_score)

                line = f"[{match_type}] {team1} vs {team2}"
                if t1_str: line += f"  |  {team1}: {t1_str}"
                if t2_str: line += f"  |  {team2}: {t2_str}"
                if venue:  line += f"  @{venue}"
                line += f"\n        → {status}"

                matches.append(line)

    if not matches:
        return "No live matches right now. Check back soon!"

    result = "🏏 LIVE & RECENT SCORES\n" + "="*50 + "\n"
    result += "\n\n".join(matches[:8])   # show max 8 matches
    return result


# ======= 2. Upcoming schedule =======
def get_upcoming_matches():
    """Returns upcoming international + domestic match schedule."""
    data = _get("matches/v1/upcoming")
    if not data:
        return "Sorry, couldn't fetch upcoming matches right now."

    matches = []
    for t in data.get("typeMatches", []):
        match_type = t.get("matchType", "")
        for series in t.get("seriesMatches", []):
            series_info = series.get("seriesAdWrapper") or series.get("adDetail")
            if not series_info:
                continue
            series_name = series_info.get("seriesName", "")
            for m in series_info.get("matches", []):
                info  = m.get("matchInfo", {})
                team1 = info.get("team1", {}).get("teamName", "?")
                team2 = info.get("team2", {}).get("teamName", "?")
                venue = info.get("venueInfo", {}).get("ground", "")
                city  = info.get("venueInfo", {}).get("city", "")
                desc  = info.get("matchDesc", "")

                line = f"[{match_type}] {team1} vs {team2}  ({desc})"
                if series_name: line += f"\n        Series: {series_name}"
                if city:        line += f"\n        📍 {venue}, {city}"

                matches.append(line)

    if not matches:
        return "No upcoming matches found."

    result = "📅 UPCOMING MATCHES\n" + "="*50 + "\n"
    result += "\n\n".join(matches[:8])
    return result


# ======= 3. Recent match results =======
def get_recent_results():
    """Returns recently completed match results."""
    data = _get("matches/v1/recent")
    if not data:
        return "Sorry, couldn't fetch recent results right now."

    matches = []
    for t in data.get("typeMatches", []):
        match_type = t.get("matchType", "")
        for series in t.get("seriesMatches", []):
            series_info = series.get("seriesAdWrapper") or series.get("adDetail")
            if not series_info:
                continue
            for m in series_info.get("matches", []):
                info   = m.get("matchInfo", {})
                score  = m.get("matchScore", {})
                team1  = info.get("team1", {}).get("teamSName", "?")
                team2  = info.get("team2", {}).get("teamSName", "?")
                status = info.get("status", "")
                city   = info.get("venueInfo", {}).get("city", "")

                t1_score = score.get("team1Score", {})
                t2_score = score.get("team2Score", {})

                def fmt(s):
                    inn = s.get("inngs1", {})
                    if not inn: return ""
                    return f"{inn.get('runs','?')}/{inn.get('wickets','?')} ({inn.get('overs','?')} ov)"

                line = f"[{match_type}] {team1} vs {team2}"
                s1, s2 = fmt(t1_score), fmt(t2_score)
                if s1: line += f"  |  {team1}: {s1}"
                if s2: line += f"  |  {team2}: {s2}"
                if city: line += f"  @{city}"
                line += f"\n        ✅ {status}"
                matches.append(line)

    if not matches:
        return "No recent results found."

    result = "📊 RECENT RESULTS\n" + "="*50 + "\n"
    result += "\n\n".join(matches[:8])
    return result


# ======= 4. Cricket news headlines =======
def get_cricket_news():
    """Returns latest cricket news headlines from Cricbuzz."""
    data = _get("news/v1/index")
    if not data:
        return "Sorry, couldn't fetch cricket news right now."

    stories = data.get("storyList", [])
    headlines = []

    for item in stories:
        story = item.get("story")
        if story:
            title     = story.get("hline", "")
            intro     = story.get("intro", "")
            timestamp = story.get("pubTime", "")
            if title:
                headlines.append(f"📰 {title}\n        {intro}")

    if not headlines:
        return "No news available right now."

    result = "📰 LATEST CRICKET NEWS\n" + "="*50 + "\n"
    result += "\n\n".join(headlines[:6])
    return result


# ======= 5. Search for a player by name =======
def search_player_live(name):
    """Search Cricbuzz for a player and return their live profile info."""
    data = _get("stats/v1/player/search", params={"plrN": name})
    if not data:
        return None

    players = data.get("plrs", [])
    if not players:
        return None

    p    = players[0]  # take top result
    pid  = p.get("id", "")
    pname = p.get("fullName", p.get("name", ""))
    country = p.get("ctryCd", "")
    role    = p.get("role", "")

    return {
        "id":      pid,
        "name":    pname,
        "country": country,
        "role":    role,
        "summary": f"{pname} ({country}) — Role: {role}"
    }