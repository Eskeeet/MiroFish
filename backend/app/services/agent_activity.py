"""Simulation activity model used by the local graph memory updater."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentActivity:
    platform: str
    agent_id: int
    agent_name: str
    action_type: str
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        return f"{self.agent_name}: {descriptions.get(self.action_type, self._describe_generic)()}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        return f"发布了一条帖子：「{content}」" if content else "发布了一条帖子"

    def _describe_like_post(self) -> str:
        content = self.action_args.get("post_content", "")
        author = self.action_args.get("post_author_name", "")
        if content and author:
            return f"点赞了{author}的帖子：「{content}」"
        if content:
            return f"点赞了一条帖子：「{content}」"
        if author:
            return f"点赞了{author}的一条帖子"
        return "点赞了一条帖子"

    def _describe_dislike_post(self) -> str:
        content = self.action_args.get("post_content", "")
        author = self.action_args.get("post_author_name", "")
        if content and author:
            return f"踩了{author}的帖子：「{content}」"
        if content:
            return f"踩了一条帖子：「{content}」"
        if author:
            return f"踩了{author}的一条帖子"
        return "踩了一条帖子"

    def _describe_repost(self) -> str:
        content = self.action_args.get("original_content", "")
        author = self.action_args.get("original_author_name", "")
        if content and author:
            return f"转发了{author}的帖子：「{content}」"
        if content:
            return f"转发了一条帖子：「{content}」"
        if author:
            return f"转发了{author}的一条帖子"
        return "转发了一条帖子"

    def _describe_quote_post(self) -> str:
        content = self.action_args.get("original_content", "")
        author = self.action_args.get("original_author_name", "")
        quote = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        if content and author:
            base = f"引用了{author}的帖子「{content}」"
        elif content:
            base = f"引用了一条帖子「{content}」"
        elif author:
            base = f"引用了{author}的一条帖子"
        else:
            base = "引用了一条帖子"
        return f"{base}，并评论道：「{quote}」" if quote else base

    def _describe_follow(self) -> str:
        target = self.action_args.get("target_user_name", "")
        return f"关注了用户「{target}」" if target else "关注了一个用户"

    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post = self.action_args.get("post_content", "")
        author = self.action_args.get("post_author_name", "")
        if not content:
            return "发表了评论"
        if post and author:
            return f"在{author}的帖子「{post}」下评论道：「{content}」"
        if post:
            return f"在帖子「{post}」下评论道：「{content}」"
        if author:
            return f"在{author}的帖子下评论道：「{content}」"
        return f"评论道：「{content}」"

    def _describe_like_comment(self) -> str:
        content = self.action_args.get("comment_content", "")
        author = self.action_args.get("comment_author_name", "")
        if content and author:
            return f"点赞了{author}的评论：「{content}」"
        if content:
            return f"点赞了一条评论：「{content}」"
        if author:
            return f"点赞了{author}的一条评论"
        return "点赞了一条评论"

    def _describe_dislike_comment(self) -> str:
        content = self.action_args.get("comment_content", "")
        author = self.action_args.get("comment_author_name", "")
        if content and author:
            return f"踩了{author}的评论：「{content}」"
        if content:
            return f"踩了一条评论：「{content}」"
        if author:
            return f"踩了{author}的一条评论"
        return "踩了一条评论"

    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"搜索了「{query}」" if query else "进行了搜索"

    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"搜索了用户「{query}」" if query else "搜索了用户"

    def _describe_mute(self) -> str:
        target = self.action_args.get("target_user_name", "")
        return f"屏蔽了用户「{target}」" if target else "屏蔽了一个用户"

    def _describe_generic(self) -> str:
        return f"执行了{self.action_type}操作"
