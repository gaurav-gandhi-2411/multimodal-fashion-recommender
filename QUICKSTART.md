# Quickstart

Get a recommendation call working in the next five minutes, using the public sandbox —
no waiting on us, no catalog handoff required yet.

## 0. The sandbox key (use this right now)

```
Brand: h_and_m
Header: X-Api-Key: hm-sandbox-demo-key
Base URL: https://fashion-recommender-staging-rm7rz66wza-el.a.run.app
```

This is a shared, public, **read-only** key against a real dataset (H&M's public Kaggle
catalog — not a paying client's data) with the same trained, validated model that serves
every brand. It exists so you can integrate and test before your own catalog is onboarded.
It is rate-limited per your IP (not shared across everyone using the key, so one noisy
caller doesn't affect anyone else) — see [Rate limits](#rate-limits) below. Don't build
production traffic against it; it's for integration testing, not for serving your users.

## 1. Your first call (curl)

Item-to-item similarity — no user history needed, works for any catalog on day one:

```bash
curl -H "X-Api-Key: hm-sandbox-demo-key" \
  "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/item/733098018/similar?k=5"
```

```json
{
  "request_id": "583f9cfa-06fc-4dbd-bb92-dc8a85ee0e3a",
  "brand": "h_and_m",
  "query_item_id": "733098018",
  "results": [
    {"item_id": "733098016", "score": 0.869, "explanation": null, "pdp_url": null},
    {"item_id": "733098009", "score": 0.798, "explanation": null, "pdp_url": null}
  ],
  "latency_ms": 1.9
}
```

Personalized recommendations — this one uses a real trained user history, not a mocked
response (validated 3.06x lift over popularity on held-out data):

```bash
curl -X POST "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/recommend" \
  -H "X-Api-Key: hm-sandbox-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "b05ebee4ae79cf2a949fac159d145728baabae5276fe900371c0423796006b39", "k": 5}'
```

`cold_start: false` in the response means this genuinely came from the trained
personalization model, not an item-similarity fallback.

## 2. The one thing that trips everyone up: CORS

**You cannot call this API directly from browser JavaScript running on your storefront.**
There is no CORS policy that allows arbitrary origins — a `fetch()` from your site's
client-side code will be blocked by the browser, full stop. This is deliberate, not an
oversight: it also keeps your API key out of client-side JS, where anyone could read it
from the page source.

You need a tiny server-side proxy — one route in whatever backend you already have. Here's
the exact pattern our own demo app uses (Next.js, but the shape is the same in any
framework):

```ts
// app/api/similar/route.ts — runs on YOUR server, not the browser
export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get("id");
  const res = await fetch(
    `https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/item/${id}/similar?k=6`,
    { headers: { "X-Api-Key": process.env.YOUR_API_KEY! } }  // server-side env var, never exposed to the browser
  );
  const data = await res.json();
  return NextResponse.json(data);
}
```

Your storefront's browser JS then calls **your own** `/api/similar` route (same-origin, no
CORS issue), which forwards to us server-side. This also means your real API key lives only
in your server's environment variables, never in a browser-visible bundle.

## 3. Response shapes you'll actually build UI around

| Endpoint | Method | Needs | Returns | Works against the `h_and_m` sandbox? |
|---|---|---|---|---|
| `/v1/{brand}/item/{id}/similar` | GET | nothing (item_id in path) | ranked list, `item_id` + `score` + `pdp_url` | ✅ yes — try it above |
| `/v1/{brand}/recommend` | POST | `{"user_id"}` or `{"item_id"}` in body | ranked list + `cold_start` flag | ✅ yes — try it above |
| `/v1/{brand}/visual-search` | POST | multipart image upload | ranked list + `match_confidence` | ✅ yes |
| `/v1/{brand}/item/{id}/complete` | GET | nothing | complementary-category items (outfit slots) | ⚠️ **200, but `enabled: false` / empty `results`** — this brand doesn't have outfit slots configured. Live on brands that do (e.g. Snitch, Powerlook). |
| `/v1/{brand}/style-search` | POST | `?text=<query>`, empty body | ranked list + `match_confidence` | ❌ **503** — this brand has no visual index built. See [§3a](#3a-style-search-a-working-example-and-why-the-obvious-curl-fails) for the exact invocation and why the "obvious" one 411s. |
| `/v1/{brand}/item/{id}/attributes` | GET | nothing | `color`, `pattern` + `reliability` tier per tag | ❌ **503** — this brand has no attribute index built. |

**Not every endpoint is wired up for every brand** — the sandbox (`h_and_m`) is the
Phase 0.5-validated two-tower model's own training brand, so `/similar` and `/recommend`
are the strongest demo of this project's actual differentiator (a validated 3.06x lift,
not a mock). `/style-search` and `/attributes` need a per-brand visual/attribute index
that the sandbox doesn't have built — a 503 there means "not configured for this brand,"
not "broken." Try those two against a brand that has them once you're onboarded (see
[Getting your own brand live](#getting-your-own-brand-live) below), or use
`GET /docs` to see the full request/response shape either way.

Full interactive schema (every field, every status code) is live right now:
**https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/docs**

### 3a. Style-search: a working example, and why the obvious curl fails

The endpoint takes its query as `?text=<query>` with no request body — so the natural
thing to try is:

```bash
# DON'T copy this one — it 411s before it even reaches our API
curl -X POST "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/style-search?text=blue+denim+jacket" \
  -H "X-Api-Key: hm-sandbox-demo-key"
```

That hits Google Cloud's load balancer requiring a `Content-Length` header on POST
requests — you'll get a generic Google `411 Length Required` HTML page, not a response
from our API at all, with nothing about API keys or brands in it. It's not you, and it's
not really us either — it's an infra layer in front of us. **Always send an explicit
(even empty) body on POST requests with no JSON payload:**

```bash
curl -X POST "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/style-search?text=blue+denim+jacket" \
  -H "X-Api-Key: hm-sandbox-demo-key" \
  -d ''
```

Against the `h_and_m` sandbox this correctly reaches our API and returns
`503 {"detail": "Style search not configured for brand 'h_and_m'..."}` — expected, per the
table above. Here's the same invocation against a brand that DOES have style-search
configured, so you can see the real shape (this uses a different brand's key, not the
public sandbox one — for illustration only):

```json
{
  "brand": "snitch",
  "query": "blue denim jacket",
  "results": [
    {"item_id": "1121", "score": 0.338, "pdp_url": "https://www.snitch.com/products/100-cotton-oversized-denim-lapel-collar-jacket-4msk8792-01"}
  ],
  "match_confidence": 0.026
}
```

## 4. Auth errors

Wrong or missing key → `401` with a message naming the exact header to send. Unknown brand
slug → `404`. If you see either, double check the header name is `X-Api-Key` (capital A, capital P — not `X-API-KEY`,
though FastAPI/Starlette normalize header casing so any casing actually works; the string
itself must match exactly).

## Rate limits

60 requests/minute, keyed by (your IP, the brand you're calling) — every caller against
the sandbox gets their own bucket, so it's safe to share the sandbox key publicly. No
tiering beyond that today.

## Getting your own brand live

The sandbox lets you build and test your integration right now. Getting your own catalog
onboarded is currently a short, coordinated process with us (not yet fully self-serve) —
see `CLIENT_ONBOARDING.md` for what we need from you and what to expect.
