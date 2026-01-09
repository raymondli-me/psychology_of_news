"""YouTube data collector using YouTube Data API v3."""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


@dataclass
class YouTubeItem:
    """A YouTube video or comment."""
    id: str
    text: str
    type: str  # 'video' or 'comment'
    video_id: str
    video_title: str
    channel_name: str
    like_count: int
    published_at: str
    url: str
    reply_count: Optional[int] = None  # Only for comments


class YouTubeCollector:
    """Collect YouTube videos and comments."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment")

        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        print("YouTube API initialized")

    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        order: str = "relevance",  # relevance, date, rating, viewCount
        published_after: Optional[str] = None  # ISO 8601 format
    ) -> List[Dict]:
        """Search for videos matching query."""
        print(f"Searching YouTube for '{query}'...")

        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=min(max_results, 50),  # API limit
                order=order,
                publishedAfter=published_after
            )
            response = request.execute()

            videos = []
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                videos.append({
                    "video_id": video_id,
                    "title": snippet["title"],
                    "description": snippet.get("description", ""),
                    "channel_name": snippet["channelTitle"],
                    "published_at": snippet["publishedAt"],
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })

            print(f"  Found {len(videos)} videos")
            return videos

        except HttpError as e:
            print(f"  Error searching videos: {e}")
            return []

    def get_video_comments(
        self,
        video_id: str,
        max_results: int = 100,
        order: str = "relevance"  # relevance, time
    ) -> List[Dict]:
        """Get comments for a specific video."""
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_results, 100),  # API limit
                order=order,
                textFormat="plainText"
            )
            response = request.execute()

            comments = []
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment_id": item["id"],
                    "text": snippet["textDisplay"],
                    "author": snippet["authorDisplayName"],
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet["publishedAt"],
                    "reply_count": item["snippet"].get("totalReplyCount", 0)
                })

            return comments

        except HttpError as e:
            # Comments might be disabled
            if "commentsDisabled" in str(e):
                print(f"    Comments disabled for video {video_id}")
            else:
                print(f"    Error getting comments: {e}")
            return []

    def search_with_comments(
        self,
        query: str,
        max_videos: int = 30,
        max_comments_per_video: int = 50,
        order: str = "relevance"
    ) -> List[YouTubeItem]:
        """Search videos and collect their comments."""
        items = []

        # Get videos
        videos = self.search_videos(query, max_results=max_videos, order=order)

        for video in videos:
            video_id = video["video_id"]
            video_title = video["title"]

            # Add video itself (title + description)
            video_text = f"{video['title']}. {video['description']}" if video['description'] else video['title']
            if len(video_text.strip()) > 20:
                items.append(YouTubeItem(
                    id=f"video_{video_id}",
                    text=video_text[:2000],  # Truncate very long descriptions
                    type="video",
                    video_id=video_id,
                    video_title=video_title,
                    channel_name=video["channel_name"],
                    like_count=0,  # Would need separate API call
                    published_at=video["published_at"],
                    url=video["url"]
                ))

            # Get comments
            print(f"  Getting comments for: {video_title[:50]}...")
            comments = self.get_video_comments(video_id, max_results=max_comments_per_video)

            for comment in comments:
                if len(comment["text"].strip()) > 20:
                    items.append(YouTubeItem(
                        id=f"comment_{comment['comment_id']}",
                        text=comment["text"][:2000],
                        type="comment",
                        video_id=video_id,
                        video_title=video_title,
                        channel_name=comment["author"],
                        like_count=comment["like_count"],
                        published_at=comment["published_at"],
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        reply_count=comment["reply_count"]
                    ))

            print(f"    Collected {len(comments)} comments")

        print(f"\nTotal collected: {len(items)} items")
        return items


def collect_youtube(
    query: str,
    max_videos: int = 30,
    max_comments: int = 50,
    output_path: Optional[str] = None
) -> List[Dict]:
    """Convenience function to collect YouTube data."""
    collector = YouTubeCollector()
    items = collector.search_with_comments(
        query=query,
        max_videos=max_videos,
        max_comments_per_video=max_comments
    )

    # Convert to dicts
    data = [asdict(item) for item in items]

    # Save if output path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} items to {output_path}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect YouTube data")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-videos", type=int, default=30)
    parser.add_argument("--max-comments", type=int, default=50)
    parser.add_argument("--output", "-o", default="youtube_data.json")

    args = parser.parse_args()

    data = collect_youtube(
        query=args.query,
        max_videos=args.max_videos,
        max_comments=args.max_comments,
        output_path=args.output
    )

    print(f"\nCollected {len(data)} total items")
