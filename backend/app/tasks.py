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

# async playwright for article scraping (optional/fallback)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# NLP / ML libs
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

import wikipediaapi


# Wordcloud + NLTK
from wordcloud import WordCloud
import nltk
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
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
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # remove urls
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # keep common punctuation but remove other weird chars
    text = re.sub(r"[^\w\s.,!?\'\"-]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_social_output(source: str, id: str, text: str, author: Optional[str], date: Optional[Any], link: Optional[str]) -> Dict[str, Any]:
    return {"source": source, "id": str(id), "author": author, "date": str(date) if date is not None else None, "text": clean_text(text or ""), "link": link}

# -----------------------------------------------------------------------------
# SECTION A: Social Media scrapers (YouTube + Reddit)
# -----------------------------------------------------------------------------

async def scrape_youtube_async(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Uses googleapiclient youtube_client (synchronous) but runs in thread.
    Returns list of normalized items.
    """
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
    """
    Uses praw reddit_client (synchronous) but runs in thread.
    Searches r/all for query and returns posts.
    """
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

import aiohttp
import feedparser
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# SECTION: WIKIPEDIA (async REST API)
# -----------------------------------------------------------------------------

async def fetch_wikipedia_data_async(topic: str, max_links: int = 3, lang: str = "id"):
    """
    Fetches summary + linked section snippets from Wikipedia using REST API.
    Fallback to English if Indonesian not found.
    """
    results = []

    async def fetch_page(session, lang_code, title):
        url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/summary/{title}"
        headers = {"User-Agent": "Mozilla/5.0"}  # <-- FIX: Added User-Agent header
        try:
            async with session.get(url, timeout=10, headers=headers) as resp: # <-- FIX: Passed headers
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
        except asyncio.TimeoutError:
            print(f"[WARN] Wikipedia fetch timeout: {title} ({lang_code})")
        except Exception as e:
            print(f"[WARN] Wikipedia fetch failed for {title} ({lang_code}): {e}")
        return None

    async with aiohttp.ClientSession() as session:
        # Try main topic (Indonesian → English)
        main_data = await fetch_page(session, lang, topic.replace(" ", "_"))
        if not main_data:
            main_data = await fetch_page(session, "en", topic.replace(" ", "_"))
        if main_data:
            results.append(main_data)

        # Fetch linked pages (use opensearch to find related pages)
        search_url = f"https://{lang}.wikipedia.org/w/api.php?action=opensearch&search={topic}&limit={max_links}&namespace=0&format=json"
        headers = {"User-Agent": "Mozilla/5.0"} # <-- FIX: Added User-Agent header
        try:
            async with session.get(search_url, timeout=10, headers=headers) as resp: # <-- FIX: Passed headers
                data = await resp.json()
                titles = data[1] if len(data) > 1 else []
                for title in titles:
                    sub_data = await fetch_page(session, lang, title.replace(" ", "_"))
                    if sub_data:
                        results.append(sub_data)
        except Exception as e:
            print(f"[WARN] Wikipedia search failed: {e}")

    return results


# -----------------------------------------------------------------------------
# SECTION: GOOGLE NEWS (feedparser + BeautifulSoup)
# -----------------------------------------------------------------------------

async def fetch_google_news_async(topic: str, max_articles: int = 5, lang: str = "id", region: str = "ID"):
    """
    Fetch Google News RSS articles (no Playwright, Windows-safe).
    """
    encoded_topic = urllib.parse.quote(topic)
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl={lang}&gl={region}&ceid={region}:{lang}"

    results = []
    feed = feedparser.parse(rss_url, request_headers={"User-Agent": "Mozilla/5.0"})
    print(f"[DEBUG] Fetched {len(feed.entries)} RSS entries for topic '{topic}'")

    async with aiohttp.ClientSession() as session:
        for entry in feed.entries[:max_articles]:
            link = entry.link
            title = entry.title
            source = getattr(getattr(entry, "source", None), "title", "Google News")
            
            # ✅ Always initialize text
            text = ""

            try:
                async with session.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as resp:
                    if resp.status != 200:
                        continue
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                    text = " ".join([p for p in paragraphs if len(p) > 50])[:4000]
            except asyncio.TimeoutError:
                print(f"[WARN] Timeout fetching article: {link}")
            except Exception as e:
                print(f"[WARN] Failed to fetch article '{title}': {e}")

            # ✅ Fallback if text is still empty
            if not text.strip():
                text = entry.get("summary", entry.get("description", ""))

            if not text.strip():
                continue

            results.append({
                "title": title,
                "section": "News",
                "content": text,
                "source": source,
                "date": getattr(entry, "published", None),
                "url": link
            })
    return results



# -----------------------------------------------------------------------------
# SECTION D: Sentiment classification (Hugging Face inference via HF router)
# -----------------------------------------------------------------------------
hf_headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"} if HUGGINGFACE_API_KEY else {}
PRIMARY_MODEL = "tabularisai/multilingual-sentiment-analysis"
 # multilingual 3-class
CHUNK_SIZE = 5  # Adjust for API limits
hf_headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


# 🔹 Function to safely query HF API in chunks with retry
async def _query_hf_model(model: str, texts: list[str]) -> list:
    url = f"https://api-inference.huggingface.co/models/{model}"
    results = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i, text in enumerate(texts, start=1):
            clean_text = text.strip()
            if not clean_text:
                print(f"[WARN] Skipping empty text #{i}")
                results.append(None)
                continue

            payload = {"inputs": clean_text}
            try:
                resp = await client.post(url, headers=hf_headers, json=payload)
                if resp.status_code == 503:
                    # model still loading — wait a bit
                    print("[INFO] Model warming up, retrying...")
                    await asyncio.sleep(5)
                    resp = await client.post(url, headers=hf_headers, json=payload)
                resp.raise_for_status()
                results.append(resp.json())
            except Exception as e:
                print(f"[ERROR] HF API failed for text #{i}: {e}")
                results.append(None)
    return results


# 🔹 Function to map HF model output → binary sentiment
def map_to_binary_label(preds: list[dict]) -> str:
    if not preds:
        return "unknown"

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    top = preds[0]
    if top["label"].lower() == "neutral":
        for alt in preds[1:]:
            if "positive" in alt["label"].lower() or "negative" in alt["label"].lower():
                top = alt
                break

    label = top["label"].lower()
    if "positive" in label:
        return "positive"
    elif "negative" in label:
        return "negative"
    return "unknown"


# 🔹 Main classification function
async def classify_sentiment_batch(texts: list[str]) -> list[dict]:
    """Classify sentiments for multiple texts and return sentiment + confidence."""
    print(f"--- Running sentiment classification for {len(texts)} texts ---")
    results = []

    for i, text in enumerate(texts, start=1):
        clean_text = (text or "").strip()
        if not clean_text:
            print(f"[WARN] Skipping text #{i}: empty or invalid")
            results.append(None)
            continue

        try:
            payload = {"inputs": clean_text}
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/tabularisai/multilingual-sentiment-analysis",
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json=payload,
                timeout=30
            )
            data = resp.json()

            # Handle retry if empty
            if not data or isinstance(data, dict):
                print(f"[WARN] Empty/invalid response for #{i}, retrying...")
                await asyncio.sleep(2)
                resp = requests.post(
                    f"https://api-inference.huggingface.co/models/tabularisai/multilingual-sentiment-analysis",
                    headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                    json=payload,
                    timeout=30
                )
                data = resp.json()

            # ✅ Extract predictions safely
            preds = None
            if isinstance(data, list):
                preds = data[0] if isinstance(data[0], list) else data
            if not preds:
                print(f"[DEBUG] Skipped text #{i}: {clean_text!r}")

            if preds and len(preds) > 0:
                binary_label = map_to_binary_label(preds)
                results.append({
                    "text": clean_text,
                    "sentiment": binary_label,
                    "confidence": max(p['score'] for p in preds)
                })
            else:
                print(f"[WARN] Skipping text #{i} due to missing sentiment")
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
    distribution = Counter(doc.get("sentiment", "neutral") for doc in docs)
    all_text = " ".join(doc.get("text", "") for doc in docs)
    words = [w for w in re.findall(r'\b\w{3,}\b', all_text.lower()) if w not in stop_words]
    top_words = Counter(words).most_common(15)
    word_cloud_image = ""
    if words:
        try:
            wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(" ".join(words[:5000]))
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
    if not hf_client:
        return "LLM client not initialized."
    if not social_results:
        return "Not enough social media data was found."
    dist = quantitative_analysis.get("distribution", {})
    total = sum(dist.values()) or 1
    dist_str = ", ".join([f"{c} {s} ({round(c/total*100)}%)" for s, c in dist.items()])
    top_words_str = ", ".join([f"'{w}'" for w, c in quantitative_analysis.get("top_words", [])])
    # Provide sample posts (top 20)
    context = "\n".join([f"{i+1}. [{r.get('source').upper()}] ({r.get('sentiment','N/A')}) {r.get('text','')[:250]}" for i, r in enumerate(social_results[:20])])
    prompt = f"""You are a sentiment analysis expert. Analyze social media posts about \"{query}\" and create a report.

**Quantitative Analysis:**
- Sentiment Distribution: {dist_str}
- Top Keywords: {top_words_str}

**Sample Social Media Posts:**
{context}

Based STRICTLY on the data provided, write a concise sentiment analysis report (300-400 words) including:
1. Overall Sentiment (use exact distribution above).
2. Key Themes from the posts and keywords.
3. A 2-3 sentence conclusion.
Do not invent reasons or sentiments not present in the data."""
    try:
        completion = await hf_client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role":"user","content":prompt}], temperature=0.5, max_tokens=600)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Sentiment report generation failed: {e}"

async def generate_reasoning_report(query: str, news_wiki_results: List[Dict[str, Any]]) -> str:
    if not hf_client:
        return "LLM client not initialized."
    if not news_wiki_results:
        return "Not enough news or articles were found."
    # Build context
    context = "\n".join([f"{i+1}. [{r.get('source')}] {r.get('title', r.get('section',''))}\n{r.get('content','')[:400]}\n" for i, r in enumerate(news_wiki_results[:30])])
    prompt = f"""You are a political/situational analysis expert. Explain the context behind discussions about \"{query}\" based on the provided news and background.

{context}

Write an analytical report (400-500 words) explaining:
1. Background Context & Recent Developments
2. Underlying Political/Social Factors
3. Different Perspectives & Implications

Base your analysis ONLY on the sources provided."""
    try:
        completion = await hf_client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=800)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Reasoning generation failed: {e}"

# -----------------------------------------------------------------------------
# SECTION G: RAG selection + Orchestration
# -----------------------------------------------------------------------------

async def rag_pipeline_and_sentiment_analysis(query: str, total_limit: int = 20, k: int = 8):
    """
    Collect social media (Reddit + YouTube) -> classify sentiments -> build embeddings -> faiss search to pick relevant docs
    Returns: relevant_docs, all_social_docs
    """
    # 1) Collect social posts
    print(f"--- Starting social scrapers for: {query}")
    try:
        yt_task = scrape_youtube_async(query, limit=total_limit//2 or total_limit)
        rd_task = scrape_reddit_async(query, limit=total_limit//2 or total_limit)
        # <-- FIX: Removed the un-awaited call to scrape_google_search_async
        resp = await asyncio.gather(yt_task, rd_task)
        all_social = []
        for sub in resp:
            all_social.extend(sub or [])
    except Exception as e:
        print(f"--- [ERROR] Social collection failed: {e}")
        traceback.print_exc()
        all_social = []

    # ensure limit
    all_social = all_social[:total_limit]

    # 2) classify sentiments in-place
    texts = [d.get("text", "") for d in all_social]
    sentiment_results = await classify_sentiment_batch(texts)

    for doc, sent in zip(all_social, sentiment_results):
        if sent:
            doc["sentiment"] = sent["sentiment"]
            doc["confidence"] = sent["confidence"]


    # 3) Prepare embeddings and FAISS
    valid_docs = [d for d in all_social if d.get("text","").strip()]
    if not valid_docs:
        return [], all_social

    if SentenceTransformer is None:
        print("⚠️ sentence-transformers not available; returning all docs as relevant fallback.")
        return valid_docs[:k], all_social

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts_for_rag = [clean_text(d.get("text","")) for d in valid_docs]
    embeddings = await asyncio.to_thread(embedder.encode, texts_for_rag, convert_to_numpy=True)

    if faiss is None:
        print("⚠️ faiss not available; returning top-k docs by naive heuristic (confidence).")
        # fallback: sort by confidence then return top k
        sorted_docs = sorted(valid_docs, key=lambda x: x.get("confidence",0.0), reverse=True)
        return sorted_docs[:k], all_social

    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        query_emb = await asyncio.to_thread(embedder.encode, [query], convert_to_numpy=True)
        k_adjusted = min(k, len(valid_docs))
        _, indices = index.search(query_emb.astype('float32'), k_adjusted)
        relevant_docs = [valid_docs[i] for i in indices[0] if i < len(valid_docs)]
    except Exception as e:
        print(f"--- [ERROR] FAISS or embedding search failed: {e}")
        traceback.print_exc()
        # fallback
        relevant_docs = valid_docs[:k]
    return relevant_docs, all_social

# Helper: google search scraping fallback (lightweight, used optionally)
from requests_html import AsyncHTMLSession
async def scrape_google_search_async(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Lightweight Google search scraping (used optionally). Not primary for sentiment/ reasoning in this architecture.
    """
    results = []
    if limit <= 0:
        return results
    session = AsyncHTMLSession()
    url = f'https://www.google.com/search?q={urllib.parse.quote_plus(query)}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = await session.get(url, headers=headers)
        r.raise_for_status()
        search_results = r.html.find('div.g')
        count = 0
        for result in search_results:
            if count >= limit:
                break
            title_elem = result.find('h3', first=True)
            link_elem = result.find('a', first=True)
            snippet_elem = result.find('div[data-sncf="2"]', first=True) or result.find('.VwiC3b', first=True)
            if title_elem and link_elem:
                title = title_elem.text or ""
                link = link_elem.attrs.get('href', '')
                snippet = snippet_elem.text if snippet_elem else ""
                combined = title + " " + snippet
                results.append(normalize_social_output("google_search", f"google_{count}", combined, None, datetime.date.today(), link))
                count += 1
    except Exception as e:
        print(f"--- [WARN] Google search scraping failed: {e}")
    finally:
        try:
            await session.close()
        except Exception:
            pass
    return results

# -----------------------------------------------------------------------------
# SECTION H: Full pipeline orchestration (used by main.py)
# -----------------------------------------------------------------------------

async def run_news_wiki_scraper_async(query: str, total_limit: int = 10) -> list:
    """
    Collects background info from Wikipedia + Google News (async, Windows-safe).
    """
    print(f"--- Starting News+Wiki scrape for: '{query}' ---")
    try:
        wiki_task = fetch_wikipedia_data_async(query, max_links=3)
        news_task = fetch_google_news_async(query, max_articles=total_limit)
        wiki_data, news_data = await asyncio.gather(wiki_task, news_task)
        combined = wiki_data + news_data
        print(f"--- Completed News+Wiki scrape: {len(combined)} items ---")
        return combined
    except Exception as e:
        print(f"--- [ERROR] News+Wiki scrape failed: {e} ---")
        return []


async def run_social_scraper_async(query: str, total_limit: int = 20) -> List[Dict[str, Any]]:
    """
    Collect social media sources (YouTube + Reddit) — this is the primary data for sentiment report
    """
    try:
        # divide between youtube and reddit roughly
        yt_limit = max(1, total_limit // 2)
        rd_limit = max(1, total_limit - yt_limit)
        yt_task = scrape_youtube_async(query, limit=yt_limit)
        rd_task = scrape_reddit_async(query, limit=rd_limit)
        yt_res, rd_res = await asyncio.gather(yt_task, rd_task)
        combined = (yt_res or []) + (rd_res or [])
        # trim and return
        return combined[:total_limit]
    except Exception as e:
        print(f"--- [ERROR] run_social_scraper_async crashed: {e}")
        traceback.print_exc()
        return []

async def run_full_analysis_pipeline(query: str, total_limit: int = 20, k: int = 8):
    """
    The main entrypoint used by your FastAPI background task.
    - Collects social media (for sentiment) and news/wiki (for reasoning).
    - Runs sentiment classification on social.
    - Performs quantitative analysis and LLM sentiment report.
    - Prepares RAG relevant docs (social) and returns assembled result dict.
    """
    print(f"\n{'='*60}\nFULL ANALYSIS PIPELINE START: {query}\n{'='*60}\n")
    # Phase 1: data collection (parallel)
    social_task = rag_pipeline_and_sentiment_analysis(query, total_limit=total_limit, k=k)  # includes collect + classify + rag
    news_wiki_task = run_news_wiki_scraper_async(query, total_limit=total_limit)
    (relevant_social_docs, all_social_docs), news_wiki_results = await asyncio.gather(social_task, news_wiki_task)

    print(f"✓ Data Collection: {len(all_social_docs)} social items, {len(news_wiki_results)} news/wiki items")
    # Phase 2: Quantitative analysis (on all social docs)
    quantitative_analysis = perform_text_analysis(all_social_docs)
    # Phase 3: Sentiment report generation (social only)
    sentiment_report = await generate_sentiment_report(query, all_social_docs, quantitative_analysis)
    # Phase 4: reasoning (news + wiki) report — not auto-generated here, left to endpoint call to conserve quota
    # Build final result object
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
    """
    Generates a chat-style response using full analysis context:
    - Sentiment report (social media summary)
    - Reasoning report (news/wiki summary)
    - All social docs (raw posts + sentiments)
    - News/wiki docs (raw content)
    """
    if not hf_client:
        return "Chat functionality is disabled (no Hugging Face client)."

    query = analysis_context.get('query', 'the topic')
    sentiment_report = analysis_context.get('sentiment_report', 'N/A')
    reasoning_report = analysis_context.get('reasoning_report', 'N/A')
    social_docs = analysis_context.get('all_social_docs', [])
    news_wiki_docs = analysis_context.get('rag2_results', [])

    # 🧠 Build detailed context
    social_examples = "\n".join([
        f"- ({d.get('sentiment', 'N/A').upper()}) {d.get('text', '')[:180]}"
        for d in social_docs[:8]
    ]) or "No social data available."

    news_examples = "\n".join([
        f"- [{d.get('source', 'Unknown')}] {d.get('title', d.get('section', ''))}: {d.get('content', '')[:180]}"
        for d in news_wiki_docs[:6]
    ]) or "No news/wiki data available."

    # 🧩 System prompt combines all sources
    system_prompt = f"""
You are an AI analyst explaining insights about "{query}".
Use the provided reports AND raw data below to answer questions factually.

📊 SENTIMENT REPORT (Summary):
{sentiment_report}

📰 REASONING REPORT (News/Wiki Summary):
{reasoning_report}

💬 SOCIAL MEDIA DATA (Posts + Sentiment):
{social_examples}

🧭 NEWS/WIKI DATA (Context):
{news_examples}

Answer **only** based on these sources.
If information is missing, say so clearly. 
Be detailed but grounded in the provided context.
"""

    # Combine with user chat history
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
    import asyncio, json, os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Query to analyze")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    res = asyncio.run(run_full_analysis_pipeline(args.q, total_limit=args.limit))
    print(json.dumps({"query": args.q, "social_count": len(res.get("all_social_docs", [])), "sentiment_report_present": bool(res.get("sentiment_report"))}, indent=2))