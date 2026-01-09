#!/usr/bin/env python3
"""Generate mock Reddit data to test the visualization without API credentials."""

import os
import sys
import json
import asyncio
import random
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.pipeline import process_source


# Mock Reddit posts and comments about Draymond Green trade
MOCK_DATA = [
    # r/nba posts
    {"text": "Draymond Green needs to be traded ASAP. The Warriors can't keep dealing with his suspensions.", "subreddit": "nba", "type": "post", "score": 1523, "url": "https://reddit.com/r/nba/1", "post_title": "Draymond needs to go"},
    {"text": "Hot take: Warriors should trade Draymond for picks and tank for a rebuild", "subreddit": "nba", "type": "post", "score": 892, "url": "https://reddit.com/r/nba/2", "post_title": "Warriors rebuild time?"},
    {"text": "Draymond's value is at an all-time low after the suspensions. No team wants that headache.", "subreddit": "nba", "type": "comment", "score": 445, "url": "https://reddit.com/r/nba/1/c1", "post_title": "Draymond needs to go"},
    {"text": "People forget Draymond is a DPOY and crucial to the Warriors system. He's not going anywhere.", "subreddit": "nba", "type": "comment", "score": 312, "url": "https://reddit.com/r/nba/1/c2", "post_title": "Draymond needs to go"},
    {"text": "The Warriors dynasty is over. Steph, Klay, and Draymond are all past their primes.", "subreddit": "nba", "type": "comment", "score": 201, "url": "https://reddit.com/r/nba/2/c1", "post_title": "Warriors rebuild time?"},
    {"text": "Steve Kerr has lost control of this team. First Draymond punching people, now this.", "subreddit": "nba", "type": "comment", "score": 567, "url": "https://reddit.com/r/nba/3/c1", "post_title": "Kerr's response"},
    {"text": "Unpopular opinion: Draymond is still elite defensively and the Warriors would be worse without him", "subreddit": "nba", "type": "post", "score": 234, "url": "https://reddit.com/r/nba/4", "post_title": "Draymond still elite?"},
    {"text": "Lakers should trade for Draymond. He'd be perfect next to AD and LeBron needs another playmaker.", "subreddit": "nba", "type": "comment", "score": 156, "url": "https://reddit.com/r/nba/5/c1", "post_title": "Trade destinations"},

    # r/warriors posts
    {"text": "I'm so tired of the national media pushing the Draymond trade narrative. He's a Warrior for life.", "subreddit": "warriors", "type": "post", "score": 892, "url": "https://reddit.com/r/warriors/1", "post_title": "Media narrative"},
    {"text": "We need to keep the core together. Trading Draymond would be a mistake we regret for years.", "subreddit": "warriors", "type": "comment", "score": 445, "url": "https://reddit.com/r/warriors/1/c1", "post_title": "Media narrative"},
    {"text": "Draymond is the heart and soul of this team. Without him we're just another team with Steph.", "subreddit": "warriors", "type": "comment", "score": 378, "url": "https://reddit.com/r/warriors/1/c2", "post_title": "Media narrative"},
    {"text": "I love Draymond but at some point we need to think about the future. Kuminga needs minutes.", "subreddit": "warriors", "type": "post", "score": 234, "url": "https://reddit.com/r/warriors/2", "post_title": "Kuminga development"},
    {"text": "The chemistry issues are real. You can see Steph is frustrated with all the drama.", "subreddit": "warriors", "type": "comment", "score": 312, "url": "https://reddit.com/r/warriors/3/c1", "post_title": "Team chemistry"},
    {"text": "Warriors front office has shown loyalty to our core. They're not trading Draymond.", "subreddit": "warriors", "type": "comment", "score": 267, "url": "https://reddit.com/r/warriors/4/c1", "post_title": "Front office stance"},
    {"text": "Draymond's contract is actually trade-friendly now. Teams looking for veteran leadership might bite.", "subreddit": "warriors", "type": "post", "score": 156, "url": "https://reddit.com/r/warriors/5", "post_title": "Contract analysis"},
    {"text": "The real question is what can we even get for Draymond? His value is tanked.", "subreddit": "warriors", "type": "comment", "score": 201, "url": "https://reddit.com/r/warriors/5/c1", "post_title": "Contract analysis"},

    # r/nbadiscussion posts
    {"text": "Analyzing the Warriors' options: A breakdown of potential Draymond Green trade scenarios", "subreddit": "nbadiscussion", "type": "post", "score": 567, "url": "https://reddit.com/r/nbadiscussion/1", "post_title": "Trade scenarios analysis"},
    {"text": "From a pure basketball standpoint, Draymond's defensive versatility is still elite. The issue is availability and temperament.", "subreddit": "nbadiscussion", "type": "comment", "score": 334, "url": "https://reddit.com/r/nbadiscussion/1/c1", "post_title": "Trade scenarios analysis"},
    {"text": "The Warriors' salary cap situation makes a Draymond trade complicated. They'd need to take back salary.", "subreddit": "nbadiscussion", "type": "comment", "score": 289, "url": "https://reddit.com/r/nbadiscussion/1/c2", "post_title": "Trade scenarios analysis"},
    {"text": "Historical comparison: Teams that traded their defensive anchor rarely recovered. See: KG from Minny, Dikembe from ATL.", "subreddit": "nbadiscussion", "type": "comment", "score": 178, "url": "https://reddit.com/r/nbadiscussion/1/c3", "post_title": "Trade scenarios analysis"},
    {"text": "The emotional component matters. Steph reportedly wants to finish his career with Draymond and Klay.", "subreddit": "nbadiscussion", "type": "post", "score": 445, "url": "https://reddit.com/r/nbadiscussion/2", "post_title": "Steph's preference"},
    {"text": "Kerr's system fundamentally relies on Draymond's playmaking. They'd need a complete offensive redesign.", "subreddit": "nbadiscussion", "type": "comment", "score": 267, "url": "https://reddit.com/r/nbadiscussion/2/c1", "post_title": "Steph's preference"},
    {"text": "The suspensions this season have been a net negative but Draymond's on-court impact remains massive.", "subreddit": "nbadiscussion", "type": "comment", "score": 223, "url": "https://reddit.com/r/nbadiscussion/3/c1", "post_title": "On-court impact"},
    {"text": "Realistic trade packages: Brooklyn has picks, Sacramento needs defense, Miami culture might help Draymond.", "subreddit": "nbadiscussion", "type": "post", "score": 312, "url": "https://reddit.com/r/nbadiscussion/4", "post_title": "Realistic packages"},
]

# Add more varied content
MORE_COMMENTS = [
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

# Fill in missing fields for MORE_COMMENTS
for i, c in enumerate(MORE_COMMENTS):
    c["url"] = f"https://reddit.com/r/{c['subreddit']}/extra/{i}"
    c["post_title"] = "Discussion thread"


async def main():
    output_dir = Path(__file__).parent.parent / "reddit" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MOCK REDDIT DATA PROCESSING")
    print("=" * 60)
    print("Generating mock data for visualization testing...")
    print("=" * 60)

    # Combine all mock data
    all_data = MOCK_DATA + MORE_COMMENTS

    # Add IDs
    for i, item in enumerate(all_data):
        item["id"] = f"mock_{i}"

    print(f"\nTotal mock items: {len(all_data)}")

    # Save raw data
    with open(output_dir / "raw_reddit.json", "w") as f:
        json.dump(all_data, f, indent=2)

    # Process through pipeline (skip actual LLM rating for testing)
    print("\nProcessing through pipeline (with mock ratings)...")

    # Generate mock ratings instead of calling LLMs
    for item in all_data:
        # Generate semi-realistic scores based on content
        text_lower = item["text"].lower()

        # Base score
        base = 5

        # Adjust based on keywords
        if any(w in text_lower for w in ["needs to be traded", "trade asap", "ship", "dealt"]):
            base += 2
        if any(w in text_lower for w in ["not going anywhere", "warrior for life", "keep", "stay"]):
            base -= 2
        if any(w in text_lower for w in ["dynasty over", "implode", "rebuild"]):
            base += 1
        if any(w in text_lower for w in ["elite", "crucial", "valuable", "dpoy"]):
            base -= 1

        # Add some variance per model
        item["gpt"] = max(1, min(10, base + random.randint(-1, 1)))
        item["claude"] = max(1, min(10, base + random.randint(-1, 1)))
        item["gemini"] = max(1, min(10, base + random.randint(-1, 1)))

    # Process (skipping rating since we added mock ratings)
    processed = await process_source(
        source_type="reddit",
        items=all_data,
        output_dir=str(output_dir),
        text_key="text",
        rate=False  # Skip rating, we already have mock scores
    )

    print("\n" + "=" * 60)
    print("MOCK DATA READY!")
    print("=" * 60)
    print(f"Processed {len(processed)} items")
    print(f"Output: {output_dir}")
    print("\nTo start the Reddit visualization:")
    print(f"  cd {Path(__file__).parent.parent}")
    print("  python reddit/server.py")
    print("  # Then visit http://localhost:8001")


if __name__ == "__main__":
    asyncio.run(main())
