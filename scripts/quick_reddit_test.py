#!/usr/bin/env python3
"""Quick test: Generate Reddit mock data by modifying existing news data format."""

import json
import random
from pathlib import Path

# Mock Reddit data
MOCK_REDDIT = [
    {"text": "Draymond Green needs to be traded ASAP. The Warriors can't keep dealing with his suspensions.", "subreddit": "nba", "type": "post", "score": 1523},
    {"text": "Hot take: Warriors should trade Draymond for picks and tank for a rebuild", "subreddit": "nba", "type": "post", "score": 892},
    {"text": "Draymond's value is at an all-time low after the suspensions. No team wants that headache.", "subreddit": "nba", "type": "comment", "score": 445},
    {"text": "People forget Draymond is a DPOY and crucial to the Warriors system. He's not going anywhere.", "subreddit": "nba", "type": "comment", "score": 312},
    {"text": "The Warriors dynasty is over. Steph, Klay, and Draymond are all past their primes.", "subreddit": "nba", "type": "comment", "score": 201},
    {"text": "Steve Kerr has lost control of this team. First Draymond punching people, now this.", "subreddit": "nba", "type": "comment", "score": 567},
    {"text": "Unpopular opinion: Draymond is still elite defensively and the Warriors would be worse without him", "subreddit": "nba", "type": "post", "score": 234},
    {"text": "Lakers should trade for Draymond. He'd be perfect next to AD and LeBron needs another playmaker.", "subreddit": "nba", "type": "comment", "score": 156},
    {"text": "I'm so tired of the national media pushing the Draymond trade narrative. He's a Warrior for life.", "subreddit": "warriors", "type": "post", "score": 892},
    {"text": "We need to keep the core together. Trading Draymond would be a mistake we regret for years.", "subreddit": "warriors", "type": "comment", "score": 445},
    {"text": "Draymond is the heart and soul of this team. Without him we're just another team with Steph.", "subreddit": "warriors", "type": "comment", "score": 378},
    {"text": "I love Draymond but at some point we need to think about the future. Kuminga needs minutes.", "subreddit": "warriors", "type": "post", "score": 234},
    {"text": "The chemistry issues are real. You can see Steph is frustrated with all the drama.", "subreddit": "warriors", "type": "comment", "score": 312},
    {"text": "Warriors front office has shown loyalty to our core. They're not trading Draymond.", "subreddit": "warriors", "type": "comment", "score": 267},
    {"text": "Draymond's contract is actually trade-friendly now. Teams looking for veteran leadership might bite.", "subreddit": "warriors", "type": "post", "score": 156},
    {"text": "The real question is what can we even get for Draymond? His value is tanked.", "subreddit": "warriors", "type": "comment", "score": 201},
    {"text": "Analyzing the Warriors' options: A breakdown of potential Draymond Green trade scenarios", "subreddit": "nbadiscussion", "type": "post", "score": 567},
    {"text": "From a pure basketball standpoint, Draymond's defensive versatility is still elite.", "subreddit": "nbadiscussion", "type": "comment", "score": 334},
    {"text": "The Warriors' salary cap situation makes a Draymond trade complicated. They'd need to take back salary.", "subreddit": "nbadiscussion", "type": "comment", "score": 289},
    {"text": "Historical comparison: Teams that traded their defensive anchor rarely recovered.", "subreddit": "nbadiscussion", "type": "comment", "score": 178},
    {"text": "The emotional component matters. Steph reportedly wants to finish his career with Draymond and Klay.", "subreddit": "nbadiscussion", "type": "post", "score": 445},
    {"text": "Kerr's system fundamentally relies on Draymond's playmaking. They'd need a complete offensive redesign.", "subreddit": "nbadiscussion", "type": "comment", "score": 267},
    {"text": "The suspensions this season have been a net negative but Draymond's on-court impact remains massive.", "subreddit": "nbadiscussion", "type": "comment", "score": 223},
    {"text": "Realistic trade packages: Brooklyn has picks, Sacramento needs defense, Miami culture might help Draymond.", "subreddit": "nbadiscussion", "type": "post", "score": 312},
    {"text": "Draymond to the Bulls makes sense. They need veteran leadership and defense badly.", "subreddit": "nba", "type": "comment", "score": 134},
    {"text": "No way the Warriors trade him mid-season. Maybe in the offseason if things don't improve.", "subreddit": "warriors", "type": "comment", "score": 289},
    {"text": "The whole 'untradeable' narrative is overblown. Every player has a price.", "subreddit": "nba", "type": "comment", "score": 167},
    {"text": "Steph would demand a trade if they shipped Draymond. The core stays together.", "subreddit": "warriors", "type": "comment", "score": 445},
    {"text": "Watching the Warriors implode in real-time is fascinating from a dynasty perspective.", "subreddit": "nba", "type": "comment", "score": 234},
    {"text": "Draymond's podcast comments about the organization aren't helping his trade value.", "subreddit": "nbadiscussion", "type": "comment", "score": 178},
    {"text": "The 25 million cap hit is actually reasonable for what Draymond brings.", "subreddit": "nbadiscussion", "type": "comment", "score": 145},
    {"text": "I think the Warriors are shopping him quietly. Too many leaks lately.", "subreddit": "nba", "type": "comment", "score": 312},
    {"text": "Kerr's comments today were interesting. He didn't shut down trade talk at all.", "subreddit": "warriors", "type": "comment", "score": 267},
    {"text": "Championship experience is undervalued. Draymond has won 4 rings.", "subreddit": "nbadiscussion", "type": "comment", "score": 189},
    {"text": "The real losers here are Warriors fans. Either way the dynasty feels over.", "subreddit": "nba", "type": "comment", "score": 356},
    {"text": "Poole trade aftermath still lingers. Team chemistry never fully recovered.", "subreddit": "warriors", "type": "comment", "score": 223},
    {"text": "Analytics say Draymond's defensive impact is still top 5 in the league.", "subreddit": "nbadiscussion", "type": "comment", "score": 156},
    {"text": "Hot take: Trading Draymond would be addition by subtraction for locker room.", "subreddit": "nba", "type": "comment", "score": 445},
    {"text": "As a neutral fan, I hope they keep him. Warriors basketball is more fun with Draymond.", "subreddit": "nba", "type": "comment", "score": 178},
]


def main():
    output_dir = Path(__file__).parent.parent / "reddit" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating mock Reddit visualization data...")

    # Generate points with random-ish 3D coordinates (clustered by subreddit)
    points = []
    subreddit_offsets = {"nba": (0, 0), "warriors": (3, 2), "nbadiscussion": (-2, 3)}

    for i, item in enumerate(MOCK_REDDIT):
        sub = item["subreddit"]
        offset = subreddit_offsets.get(sub, (0, 0))

        # Generate mock scores
        text_lower = item["text"].lower()
        base = 5
        if any(w in text_lower for w in ["needs to be traded", "trade asap", "ship", "dealt"]):
            base += 2
        if any(w in text_lower for w in ["not going anywhere", "warrior for life", "keep", "stay"]):
            base -= 2
        if any(w in text_lower for w in ["dynasty over", "implode", "rebuild"]):
            base += 1
        if any(w in text_lower for w in ["elite", "crucial", "valuable", "dpoy"]):
            base -= 1

        gpt = max(1, min(10, base + random.randint(-1, 1)))
        claude = max(1, min(10, base + random.randint(-1, 1)))
        gemini = max(1, min(10, base + random.randint(-1, 1)))

        # Map subreddit to cluster
        cluster_map = {"nba": 0, "warriors": 1, "nbadiscussion": 2}

        point = {
            "id": i,
            "sentence": item["text"],
            "x": offset[0] + random.uniform(-1.5, 1.5),
            "y": offset[1] + random.uniform(-1.5, 1.5),
            "z": random.uniform(-1, 1),
            "cluster": cluster_map.get(sub, -1),
            "gpt": gpt,
            "claude": claude,
            "gemini": gemini,
            "mean": (gpt + claude + gemini) / 3,
            "source": f"r/{sub}",
            "subreddit": sub,
            "type": item["type"],
            "score": item["score"],
            "url": f"https://reddit.com/r/{sub}/comments/{i}",
            "post_title": "Trade Discussion Thread"
        }
        points.append(point)

    # Save points_data.json
    with open(output_dir / "points_data.json", "w") as f:
        json.dump(points, f, indent=2)

    # Generate topic names
    topic_names = {
        "gpt": {"0": "r/nba Opinions", "1": "Warriors Fans", "2": "Trade Analysis"},
        "claude": {"0": "r/nba Opinions", "1": "Warriors Fans", "2": "Trade Analysis"},
        "gemini": {"0": "r/nba Opinions", "1": "Warriors Fans", "2": "Trade Analysis"}
    }
    with open(output_dir / "topic_names.json", "w") as f:
        json.dump(topic_names, f, indent=2)

    # Generate cluster stats
    cluster_stats = {}
    for cluster_id in [0, 1, 2]:
        cluster_points = [p for p in points if p["cluster"] == cluster_id]
        if cluster_points:
            cx = sum(p["x"] for p in cluster_points) / len(cluster_points)
            cy = sum(p["y"] for p in cluster_points) / len(cluster_points)
            cz = sum(p["z"] for p in cluster_points) / len(cluster_points)
            cluster_stats[str(cluster_id)] = {
                "count": len(cluster_points),
                "centroid": {"x": cx, "y": cy, "z": cz},
                "gpt_mean": sum(p["gpt"] for p in cluster_points) / len(cluster_points),
                "claude_mean": sum(p["claude"] for p in cluster_points) / len(cluster_points),
                "gemini_mean": sum(p["gemini"] for p in cluster_points) / len(cluster_points)
            }

    with open(output_dir / "cluster_stats.json", "w") as f:
        json.dump(cluster_stats, f, indent=2)

    print(f"Generated {len(points)} points")
    print(f"Output saved to: {output_dir}")
    print("\nTo start Reddit visualization:")
    print("  python reddit/server.py")
    print("  # Visit http://localhost:8001")


if __name__ == "__main__":
    main()
