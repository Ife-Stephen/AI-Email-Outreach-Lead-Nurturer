# tools.py
from langchain_core.tools import tool
import pandas as pd

@tool
def generate_social_post(platform: str, topic: str, tone: str = "friendly") -> str:
    """
    Generate a platform-optimized social media post.

    Args:
        platform: e.g., "Instagram", "LinkedIn", "X/Twitter", "TikTok"
        topic: The main content/topic of the post
        tone: Optional, e.g., "friendly", "professional", "humorous"

    Returns:
        A string with the post content, caption, hashtags, and suggested posting schedule.
    """
    return (
        f"Platform: {platform}\n"
        f"Topic: {topic}\n"
        f"Tone: {tone}\n\n"
        f"Post Content: [AI-generated post here]\n"
        f"Hashtags: #[topic.replace(' ', '')] #[platform] #[marketing]\n"
        f"Suggested Posting Time: 9 AM local time"
    )

@tool
def generate_hashtags(topic: str, platform: str = "Instagram") -> str:
    """
    Generate relevant hashtags for a given topic and platform.
    """
    base_hashtags = ["#socialmedia", f"#{topic.replace(' ', '')}", f"#{platform}"]
    return " ".join(base_hashtags)

@tool
def suggest_post_schedule(platform: str) -> str:
    """
    Suggest the best posting times for a given social media platform.
    """
    schedule = {
        "Instagram": "Mon/Wed/Fri at 9 AM",
        "LinkedIn": "Tue/Thu at 10 AM",
        "X/Twitter": "Mon-Fri at 8 AM & 6 PM",
        "TikTok": "Tue/Thu/Sat at 7 PM"
    }
    return schedule.get(platform, "Anytime during peak hours")

# Tool registry
TOOLS = {
    "generate_social_post": generate_social_post,
    "generate_hashtags": generate_hashtags,
    "suggest_post_schedule": suggest_post_schedule,
}
