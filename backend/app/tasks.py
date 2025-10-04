# tasks.py
import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import re
import html
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from bs4 import BeautifulSoup
import traceback
from collections import Counter
import io
import base64
from typing import List, Dict, Any, Optional
import requests
import aiohttp
import feedparser


# Wordcloud + NLTK
from wordcloud import WordCloud
import nltk
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords', quiet=True) # Added quiet=True to avoid verbose logs
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Local dependencies (clients + API keys) — keep same names as dependencies.py
from .dependencies import youtube_client, reddit_client, hf_client, HUGGINGFACE_API_KEY



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r"[^\w\s.,!?\'\"-]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_social_output(source: str, id: str, text: str, author: Optional[str], date: Optional[Any], link: Optional[str]) -> Dict[str, Any]:
    return {"source": source, "id": str(id), "author": author, "date": str(date) if date is not None else None, "text": clean_text(text or ""), "link": link}

# -----------------------------------------------------------------------------
# SECTION A: Social Media scrapers (YouTube + Reddit)
# -----------------------------------------------------------------------------

async def scrape_youtube_async(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    results = []
    if youtube_client is None:
        print("⚠️ YouTube client is not initialized. Skipping YouTube scraping.")
        return results

    def run_search():
        try:
            request = youtube_client.search().list(q=query, part="snippet", type="video", maxResults=min(50, limit))
            return request.execute()
        except Exception as e:
            print(f"--- [ERROR] YouTube API call failed: {e} ---")
            traceback.print_exc()
            return None

    try:
        resp = await asyncio.to_thread(run_search)
        if not resp:
            return results
        for item in resp.get("items", [])[:limit]:
            try:
                sn = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId") or item.get("id")
                title = sn.get("title", "")
                description = sn.get("description", "")
                channel = sn.get("channelTitle")
                published = sn.get("publishedAt")
                link = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
                text = f"{title} {description}"
                results.append(normalize_social_output("youtube", video_id or title, text, channel, published, link))
            except Exception as e:
                print(f"--- [WARN] parse youtube item failed: {e}")
        return results
    except Exception as e:
        print(f"--- [ERROR] scrape_youtube_async crashed: {e} ---")
        traceback.print_exc()
        return results

async def scrape_reddit_async(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    results = []
    if reddit_client is None:
        print("⚠️ Reddit client is not initialized. Skipping Reddit scraping.")
        return results

    def run_search():
        try:
            subreddit = reddit_client.subreddit("all")
            return list(subreddit.search(query, limit=limit, sort="relevance"))
        except Exception as e:
            print(f"--- [ERROR] Reddit search failed: {e} ---")
            traceback.print_exc()
            return []

    try:
        posts = await asyncio.to_thread(run_search)
        for p in posts:
            try:
                title = getattr(p, "title", "")
                selftext = getattr(p, "selftext", "")
                author = getattr(p.author, "name", None) if getattr(p, "author", None) else None
                created = getattr(p, "created_utc", None)
                created_dt = datetime.datetime.fromtimestamp(created) if created else None
                link = getattr(p, "url", None)
                text = f"{title} {selftext}"
                results.append(normalize_social_output("reddit", getattr(p, "id", title), text, author, created_dt, link))
            except Exception as e:
                print(f"--- [WARN] parse reddit submission failed: {e}")
        return results
    except Exception as e:
        print(f"--- [ERROR] scrape_reddit_async crashed: {e} ---")
        traceback.print_exc()
        return results

# -----------------------------------------------------------------------------
# SECTION: WIKIPEDIA (async REST API)
# -----------------------------------------------------------------------------

async def fetch_wikipedia_data_async(topic: str, max_links: int = 3, lang: str = "id"):
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SentimentAnalysisApp/1.0)"}

    async def fetch_page(session, lang_code, title):
        url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        try:
            async with session.get(url, timeout=10, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    extract = data.get("extract", "")
                    if extract.strip():
                        return {
                            "title": data.get("title", title),
                            "section": "Summary",
                            "content": extract,
                            "source": f"Wikipedia ({lang_code})",
                            "date": None,
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                        }
        except Exception as e:
            print(f"[WARN] Wikipedia fetch failed for {title} ({lang_code}): {e}")
        return None

    async with aiohttp.ClientSession() as session:
        main_data = await fetch_page(session, lang, topic)
        if not main_data:
            main_data = await fetch_page(session, "en", topic)
        if main_data:
            results.append(main_data)

        search_url = f"https://{lang}.wikipedia.org/w/api.php?action=opensearch&search={topic}&limit={max_links}&namespace=0&format=json"
        try:
            async with session.get(search_url, timeout=10, headers=headers) as resp:
                data = await resp.json()
                titles = data[1] if len(data) > 1 else []
                for title in titles:
                    if title.lower() != topic.lower():
                        sub_data = await fetch_page(session, lang, title)
                        if sub_data:
                            results.append(sub_data)
        except Exception as e:
            print(f"[WARN] Wikipedia search failed: {e}")

    return results

# -----------------------------------------------------------------------------
# SECTION: GOOGLE NEWS (feedparser + BeautifulSoup)
# -----------------------------------------------------------------------------

async def fetch_google_news_async(topic: str, max_articles: int = 5, lang: str = "id", region: str = "ID"):
    encoded_topic = urllib.parse.quote(topic)
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl={lang}&gl={region}&ceid={region}:{lang}"
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SentimentAnalysisApp/1.0)"}
    
    try:
        feed = await asyncio.to_thread(feedparser.parse, rss_url, request_headers=headers)
    except Exception as e:
        print(f"[ERROR] Failed to parse RSS feed: {e}")
        return []

    async with aiohttp.ClientSession() as session:
        for entry in feed.entries[:max_articles]:
            text = entry.get("summary", entry.get("description", ""))
            results.append({
                "title": entry.title,
                "section": "News",
                "content": clean_text(text),
                "source": getattr(getattr(entry, "source", None), "title", "Google News"),
                "date": getattr(entry, "published", None),
                "url": entry.link
            })
    return results

# -----------------------------------------------------------------------------
# SECTION D: Sentiment classification (Hugging Face inference via API)
# -----------------------------------------------------------------------------
hf_headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"} if HUGGINGFACE_API_KEY else {}

def map_to_binary_label(preds: list[dict]) -> str:
    if not preds: return "unknown"
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    top = preds[0]
    if top["label"].lower() == "neutral" and len(preds) > 1:
        top = preds[1]
    label = top["label"].lower()
    if "positive" in label: return "positive"
    if "negative" in label: return "negative"
    return "unknown"

async def classify_sentiment_batch(texts: list[str]) -> list[dict]:
    print(f"--- Running sentiment classification for {len(texts)} texts ---")
    results = []
    url = "https://api-inference.huggingface.co/models/tabularisai/multilingual-sentiment-analysis"

    async with httpx.AsyncClient(timeout=45.0) as client:
        for i, text in enumerate(texts, start=1):
            clean_text = (text or "").strip()
            if not clean_text:
                results.append(None)
                continue
            
            payload = {"inputs": clean_text}
            try:
                resp = await client.post(url, headers=hf_headers, json=payload)
                if resp.status_code == 503:
                    await asyncio.sleep(5)
                    resp = await client.post(url, headers=hf_headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

                preds = data[0] if (data and isinstance(data[0], list)) else data
                if preds and len(preds) > 0:
                    binary_label = map_to_binary_label(preds)
                    results.append({"text": clean_text, "sentiment": binary_label, "confidence": max(p['score'] for p in preds)})
                else:
                    results.append(None)
            except Exception as e:
                print(f"[ERROR] HF API failed for text #{i}: {e}")
                results.append(None)
    return results

# -----------------------------------------------------------------------------
# SECTION E: Quantitative analysis (distribution, top words, wordcloud)
# -----------------------------------------------------------------------------

def perform_text_analysis(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not docs:
        return {"distribution": {}, "top_words": [], "word_cloud": ""}
    
    # Filter out docs with unknown sentiment for a cleaner distribution chart
    filtered_docs = [doc for doc in docs if doc.get("sentiment") and doc.get("sentiment") != "unknown"]
    distribution = Counter(doc.get("sentiment") for doc in filtered_docs)
    
    all_text = " ".join(doc.get("text", "") for doc in docs)
    words = [w for w in re.findall(r'\b\w{3,}\b', all_text.lower()) if w not in stop_words]
    top_words = Counter(words).most_common(15)
    word_cloud_image = ""
    if words:
        try:
            wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(" ".join(words))
            buf = io.BytesIO()
            wc.to_image().save(buf, format='PNG')
            word_cloud_image = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception as e:
            print(f"--- [ERROR] Word cloud generation failed: {e}")
    return {"distribution": dict(distribution), "top_words": top_words, "word_cloud": word_cloud_image}

# -----------------------------------------------------------------------------
# SECTION F: LLM report generation (sentiment + reasoning)
# -----------------------------------------------------------------------------

async def generate_sentiment_report(query: str, social_results: List[Dict[str, Any]], quantitative_analysis: Dict[str, Any]) -> str:
    if not hf_client: return "LLM client not initialized."
    if not social_results: return "Not enough social media data was found to generate a report."
    
    dist = quantitative_analysis.get("distribution", {})
    total = sum(dist.values()) or 1
    dist_str = ", ".join([f"{c} {s} ({round(c/total*100)}%)" for s, c in dist.items()])
    top_words_str = ", ".join([f"'{w}'" for w, c in quantitative_analysis.get("top_words", [])])
    context = "\n".join([f"- [{r.get('source').upper()}] ({r.get('sentiment','N/A')}) {r.get('text','')[:200]}" for i, r in enumerate(social_results[:15])])
    prompt = f"""As a sentiment analyst, write a concise report (250-350 words) on social media discussions about "{query}".
STRICTLY use only the data below.

**Analysis Data:**
- Sentiment Distribution: {dist_str}
- Top Keywords: {top_words_str}

**Sample Posts:**
{context}

Your report must include:
1.  **Overall Sentiment:** State the dominant sentiment based on the distribution.
2.  **Key Themes:** Identify major topics from the posts and keywords.
3.  **Conclusion:** A brief 2-sentence summary.

Do not invent information. Your analysis must be grounded in the provided data."""
    try:
        completion = await hf_client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role":"user","content":prompt}], temperature=0.5, max_tokens=600)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Sentiment report generation failed: {e}"

async def generate_reasoning_report(query: str, news_wiki_results: List[Dict[str, Any]]) -> str:
    if not hf_client: return "LLM client not initialized."
    if not news_wiki_results: return "Not enough news or articles were found to generate a report."

    context = "\n".join([f"- SOURCE: {r.get('source')}\n  CONTENT: {r.get('content','')[:400]}\n" for r in news_wiki_results[:15]])
    prompt = f"""As an analyst, explain the context behind discussions about "{query}" using ONLY the provided news and articles.

**Sources:**
{context}

Write a report (300-400 words) explaining:
1.  **Background:** What are the key events or context?
2.  **Key Factors:** What are the underlying social or political drivers?
3.  **Conclusion:** A brief summary of the situation.

Base your analysis ONLY on the text provided. Do not add external information."""
    try:
        completion = await hf_client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=800)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Reasoning generation failed: {e}"

# -----------------------------------------------------------------------------
# SECTION G: RAG selection + Orchestration (MODIFIED FOR LOW MEMORY)
# -----------------------------------------------------------------------------

async def rag_pipeline_and_sentiment_analysis(query: str, total_limit: int = 20, k: int = 8):
    """
    MODIFIED FOR LOW MEMORY:
    - Collects social media posts.
    - Classifies sentiments via API.
    - Selects relevant documents by sorting by sentiment confidence instead of using local ML models.
    """
    # 1) Collect social posts
    print(f"--- Starting social scrapers for: {query}")
    try:
        yt_task = scrape_youtube_async(query, limit=total_limit//2 or total_limit)
        rd_task = scrape_reddit_async(query, limit=total_limit//2 or total_limit)
        resp = await asyncio.gather(yt_task, rd_task)
        all_social = [item for sublist in resp for item in sublist]
    except Exception as e:
        print(f"--- [ERROR] Social collection failed: {e}")
        all_social = []

    all_social = all_social[:total_limit]

    # 2) Classify sentiments in-place
    texts = [d.get("text", "") for d in all_social]
    sentiment_results = await classify_sentiment_batch(texts)

    for doc, sent in zip(all_social, sentiment_results):
        if sent:
            doc["sentiment"] = sent["sentiment"]
            doc["confidence"] = sent.get("confidence", 0.0)

    # 3) --- MEMORY OPTIMIZATION ---
    # Instead of using sentence-transformers and faiss, we use a lightweight relevance heuristic.
    # We sort the documents by their sentiment confidence score, assuming more confident
    # classifications are more relevant to the core topic.
    valid_docs = [d for d in all_social if d.get("text","").strip()]
    if not valid_docs:
        return [], all_social

    print("✅ Using lightweight relevance sorting (no local embeddings).")
    sorted_docs = sorted(valid_docs, key=lambda x: x.get("confidence", 0.0), reverse=True)
    relevant_docs = sorted_docs[:k]
    
    return relevant_docs, all_social

# -----------------------------------------------------------------------------
# SECTION H: Full pipeline orchestration (used by main.py)
# -----------------------------------------------------------------------------

async def run_news_wiki_scraper_async(query: str, total_limit: int = 10) -> list:
    print(f"--- Starting News+Wiki scrape for: '{query}' ---")
    try:
        wiki_task = fetch_wikipedia_data_async(query, max_links=3)
        news_task = fetch_google_news_async(query, max_articles=total_limit)
        wiki_data, news_data = await asyncio.gather(wiki_task, news_task)
        return (wiki_data or []) + (news_data or [])
    except Exception as e:
        print(f"--- [ERROR] News+Wiki scrape failed: {e} ---")
        return []

async def run_social_scraper_async(query: str, total_limit: int = 20) -> List[Dict[str, Any]]:
    try:
        yt_limit = max(1, total_limit // 2)
        rd_limit = max(1, total_limit - yt_limit)
        yt_task = scrape_youtube_async(query, limit=yt_limit)
        rd_task = scrape_reddit_async(query, limit=rd_limit)
        yt_res, rd_res = await asyncio.gather(yt_task, rd_task)
        return (yt_res or []) + (rd_res or [])
    except Exception as e:
        print(f"--- [ERROR] run_social_scraper_async crashed: {e}")
        return []

async def run_full_analysis_pipeline(query: str, total_limit: int = 20, k: int = 8):
    print(f"\n{'='*60}\nFULL ANALYSIS PIPELINE START: {query}\n{'='*60}\n")
    social_task = rag_pipeline_and_sentiment_analysis(query, total_limit=total_limit, k=k)
    news_wiki_task = run_news_wiki_scraper_async(query, total_limit=10) # Reduced limit
    (relevant_social_docs, all_social_docs), news_wiki_results = await asyncio.gather(social_task, news_wiki_task)

    print(f"✓ Data Collection: {len(all_social_docs)} social items, {len(news_wiki_results)} news/wiki items")
    quantitative_analysis = perform_text_analysis(all_social_docs)
    sentiment_report = await generate_sentiment_report(query, all_social_docs, quantitative_analysis)
    
    result = {
        "query": query,
        "sentiment_report": sentiment_report,
        "rag1_results": relevant_social_docs,
        "rag2_results": news_wiki_results,
        "reasoning_report": None,
        "chat_history": [],
        "quantitative_analysis": quantitative_analysis,
        "all_social_docs": all_social_docs
    }
    print(f"\n{'='*60}\nPIPELINE COMPLETED: {query}\n{'='*60}\n")
    return result

# -----------------------------------------------------------------------------
# SECTION I: Chat Response (LLM Q&A using reports)
# -----------------------------------------------------------------------------

async def generate_chat_response(analysis_context: dict, chat_history: list) -> str:
    if not hf_client: return "Chat functionality is disabled (no Hugging Face client)."

    query = analysis_context.get('query', 'the topic')
    sentiment_report = analysis_context.get('sentiment_report', 'N/A')
    reasoning_report = analysis_context.get('reasoning_report', 'N/A')
    
    system_prompt = f"""You are an AI analyst explaining insights about "{query}".
Use the provided reports to answer questions factually.

📊 SENTIMENT REPORT (Summary of Social Media):
{sentiment_report}

📰 REASONING REPORT (Summary of News/Wiki):
{reasoning_report}

Answer the user's question based **only** on these reports.
If the information is not in the reports, state that clearly.
Be concise and stay grounded in the provided context."""

    messages = [{"role": "system", "content": system_prompt}] + chat_history

    try:
        completion = await hf_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            temperature=0.6,
            max_tokens=700
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"I'm sorry, I encountered an error while generating the chat response: {e}"

# -----------------------------------------------------------------------------
# If run as script for quick debugging
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json, os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Query to analyze")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    res = asyncio.run(run_full_analysis_pipeline(args.q, total_limit=args.limit))
    print(json.dumps({
        "query": args.q, 
        "social_count": len(res.get("all_social_docs", [])), 
        "sentiment_report_present": bool(res.get("sentiment_report"))
    }, indent=2))
