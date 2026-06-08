# Demo Script — Indian Brand Fashion Recommender

**Who:** Salesperson + Indian fashion brand (Founder/CTO/Head of Growth)
**Time:** ~20 minutes
**Goal:** Show that content-based "more like this" works on their catalog today, and that personalized recommendations improve as they share traffic data.

---

## Pre-meeting Setup (5 min before)

- [ ] Run `uvicorn app.main:app --reload` — confirm `/health` returns 200
- [ ] Run `streamlit run app/streamlit_app.py` — confirm the app loads
- [ ] In the brand dropdown, select the brand matching the client (Snitch for men's streetwear, Fashor for ethnic wear, Powerlook for smart-casual)
- [ ] Have the demo_script open in a separate window for reference

---

## Opening (2 min)

> "Let me show you how the recommender works on your own catalog — real products, real images, real links to your store."

Select the client's brand from the dropdown. The UI white-labels to their brand name.

---

## Demo Flow A — "More Like This" (8 min) <- Lead with this

This is the hero story. Content-based retrieval transfers cleanly to any new catalog.

1. In the "More Like This" tab, pick a seed item from the dropdown:
   - For Snitch: pick an oversized shirt or graphic tee
   - For Fashor: pick a "3P Kurta Set" (three-piece set — the site's bestseller category)
   - For Powerlook: pick a linen shirt

2. Click **Find Similar**.

3. Walk through the results:
   > "Notice the recommendations stay in the same style family — similar silhouette, similar material. The model looks at both the product image and the description, so it catches visual similarity that text alone would miss."

4. Click a "View on site" link → the client's actual PDP opens.
   > "These link directly to your store. Everything the customer sees in the recommendation takes them to a real product page."

5. Try a second seed item in a different category to show breadth.

**Key talking points:**
- Works on day one — no interaction data required
- Both image and text signals used (CLIP for visual style, SBERT for product category/fabric)
- Indian fashion vocabulary: kurta, ethnic, indo-western — the model names them correctly
- Real PDP links → zero engineering effort to plug into your product page

---

## Demo Flow B — Personalized Recommendations (5 min) <- Illustrative

Switch to the "Personalized Recommendations" tab.

> "This tab shows what personalized recommendations look like — we're using synthetic demo users to illustrate the API surface. In production, this uses your actual shoppers' purchase history."

1. Pick any demo user from the dropdown (labelled "(synthetic demo user)").

2. Click **Get Recommendations**.

3. Walk through results briefly.
   > "The personalization model learns from what each user has bought — over time it picks up on their style, their price point, whether they prefer ethnic or western. The recommendations improve as more of your traffic flows through."

**Key talking points:**
- The API surface is the same whether user data is synthetic or real
- Personalization improves as brand interaction data accumulates (30-day ramp typical)
- The honest story: content recs are strong day one; personalization is the Phase 2 value

---

## Q&A Prep

**"How do you know your model works on our catalog?"**
> "The item tower — the part that understands what a product looks like and what it is — was trained on a large fashion dataset and transfers directly. CLIP has seen ethnic wear, saris, kurtas. The 'more like this' you just saw uses that same model, on your exact catalog."

**"What data do you need from us?"**
> "For content-based recs: just your product catalog — product ID, title, description, image URL, price, category, and product page URL. We can ingest a CSV or connect to Shopify directly. For personalized recs: we'd need a feed of purchase/click events, ideally with timestamps. The more history you share, the better the personalization."

**"How long to go live?"**
> "Content recs: under 30 minutes from catalog CSV to live API. Personalization: depends on how much interaction data you have — typically 30 days of traffic to see meaningful improvement."

**"Can we white-label it?"**
> "Yes — the API is multi-tenant. Each brand gets its own namespace, its own API key, and the demo already shows your brand name and links to your store."

---

## Close

> "The next step is a pilot: you share a catalog export, we ingest it in under 30 minutes, and you see your own products in the recommender. From there, we instrument the API on one page — 'similar items' on the PDP — and start collecting signal. Three months of traffic and you have real personalization."

---
*Built on: CLIP + SBERT + Two-Tower Transformer · FAISS retrieval · Multi-tenant FastAPI · Streamlit demo*
