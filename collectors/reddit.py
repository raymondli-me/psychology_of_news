"""Reddit data collector using PRAW."""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import praw
from praw.models import Submission, Comment


@dataclass
class RedditItem:
    """A Reddit post or comment."""
    id: str
    text: str
    type: str  # 'post' or 'comment'
    subreddit: str
    post_title: str
    author: str
    score: int
    created_utc: float
    url: str
    parent_id: Optional[str] = None


class RedditCollector:
    """Collect Reddit posts and comments."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "psychology_of_news:v1.0 (research)"
    ):
        self.reddit = praw.Reddit(
            client_id=client_id or os.getenv("REDDIT_CLIENT_ID"),
            client_secret=client_secret or os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=user_agent
        )
        print(f"Reddit API initialized (read-only: {self.reddit.read_only})")

    def search(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        max_posts: int = 50,
        max_comments_per_post: int = 20,
        time_filter: str = "month",  # hour, day, week, month, year, all
        sort: str = "relevance"  # relevance, hot, top, new, comments
    ) -> List[RedditItem]:
        """Search Reddit for posts and comments matching query."""
        items = []

        # Default subreddits for NBA content
        if subreddits is None:
            subreddits = ["nba", "warriors", "nbadiscussion"]

        for subreddit_name in subreddits:
            print(f"\nSearching r/{subreddit_name} for '{query}'...")

            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts = subreddit.search(
                    query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=max_posts
                )

                post_count = 0
                for post in posts:
                    post_count += 1

                    # Add post itself if it has text
                    post_text = f"{post.title}. {post.selftext}" if post.selftext else post.title
                    if len(post_text.strip()) > 20:  # Skip very short posts
                        items.append(RedditItem(
                            id=f"post_{post.id}",
                            text=post_text,
                            type="post",
                            subreddit=subreddit_name,
                            post_title=post.title,
                            author=str(post.author) if post.author else "[deleted]",
                            score=post.score,
                            created_utc=post.created_utc,
                            url=f"https://reddit.com{post.permalink}"
                        ))

                    # Get comments
                    post.comments.replace_more(limit=0)  # Skip "load more" links
                    comment_count = 0

                    for comment in post.comments.list()[:max_comments_per_post]:
                        if isinstance(comment, Comment) and comment.body and len(comment.body) > 20:
                            if comment.body not in ["[deleted]", "[removed]"]:
                                items.append(RedditItem(
                                    id=f"comment_{comment.id}",
                                    text=comment.body,
                                    type="comment",
                                    subreddit=subreddit_name,
                                    post_title=post.title,
                                    author=str(comment.author) if comment.author else "[deleted]",
                                    score=comment.score,
                                    created_utc=comment.created_utc,
                                    url=f"https://reddit.com{comment.permalink}",
                                    parent_id=comment.parent_id
                                ))
                                comment_count += 1

                print(f"  Found {post_count} posts, collected {sum(1 for i in items if i.subreddit == subreddit_name)} items")

            except Exception as e:
                print(f"  Error searching r/{subreddit_name}: {e}")

        print(f"\nTotal collected: {len(items)} items")
        return items

    def search_all(
        self,
        query: str,
        max_posts: int = 100,
        max_comments_per_post: int = 30,
        time_filter: str = "month"
    ) -> List[RedditItem]:
        """Search all of Reddit (not limited to specific subreddits)."""
        items = []
        print(f"Searching all of Reddit for '{query}'...")

        all_subreddit = self.reddit.subreddit("all")
        posts = all_subreddit.search(
            query,
            sort="relevance",
            time_filter=time_filter,
            limit=max_posts
        )

        for post in posts:
            post_text = f"{post.title}. {post.selftext}" if post.selftext else post.title
            if len(post_text.strip()) > 20:
                items.append(RedditItem(
                    id=f"post_{post.id}",
                    text=post_text,
                    type="post",
                    subreddit=post.subreddit.display_name,
                    post_title=post.title,
                    author=str(post.author) if post.author else "[deleted]",
                    score=post.score,
                    created_utc=post.created_utc,
                    url=f"https://reddit.com{post.permalink}"
                ))

            post.comments.replace_more(limit=0)
            for comment in post.comments.list()[:max_comments_per_post]:
                if isinstance(comment, Comment) and comment.body:
                    if len(comment.body) > 20 and comment.body not in ["[deleted]", "[removed]"]:
                        items.append(RedditItem(
                            id=f"comment_{comment.id}",
                            text=comment.body,
                            type="comment",
                            subreddit=post.subreddit.display_name,
                            post_title=post.title,
                            author=str(comment.author) if comment.author else "[deleted]",
                            score=comment.score,
                            created_utc=comment.created_utc,
                            url=f"https://reddit.com{comment.permalink}",
                            parent_id=comment.parent_id
                        ))

        print(f"Total collected: {len(items)} items")
        return items


def collect_reddit(
    query: str,
    subreddits: Optional[List[str]] = None,
    max_posts: int = 50,
    max_comments: int = 20,
    time_filter: str = "month",
    output_path: Optional[str] = None
) -> List[Dict]:
    """Convenience function to collect Reddit data."""
    collector = RedditCollector()
    items = collector.search(
        query=query,
        subreddits=subreddits,
        max_posts=max_posts,
        max_comments_per_post=max_comments,
        time_filter=time_filter
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
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Collect Reddit data")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--subreddits", nargs="+", default=["nba", "warriors"])
    parser.add_argument("--max-posts", type=int, default=50)
    parser.add_argument("--max-comments", type=int, default=20)
    parser.add_argument("--time", default="month", choices=["hour", "day", "week", "month", "year", "all"])
    parser.add_argument("--output", "-o", default="reddit_data.json")

    args = parser.parse_args()

    data = collect_reddit(
        query=args.query,
        subreddits=args.subreddits,
        max_posts=args.max_posts,
        max_comments=args.max_comments,
        time_filter=args.time,
        output_path=args.output
    )

    print(f"\nCollected {len(data)} total items")
