"""Portföy simülasyonu - tek dosya. Ayarlar en üstteki class C içinde."""
import os, json, uuid, types
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests, yfinance as yf

class C:

    INITIAL_CASH = 100_000          # TL, sanal başlangıç sermayesi
    CASH_RESERVE_PCT = 0.20         # kasada her zaman toplam değerin en az %20'si nakit
    MAX_POSITION_PCT = 0.15         # tek varlığa en fazla %15
    MIN_TRADE_TL = 1_000            # bundan küçük işlem yapma
    MAX_PROPOSALS_PER_RUN = 3

    BUY_SCORE_THRESHOLD = 70        # bu skorun altında sessiz bekle
    SELL_SCORE_BELOW = 35           # skor bunun altına düşerse sat
    STOP_LOSS = -0.08               # %8 zarar -> sat
    TAKE_PROFIT = 0.20              # %20 kâr -> sat
    TRAILING_STOP = 0.07            # zirveden %7 düşüş -> sat (kâr varsa)

    PROPOSAL_TTL_HOURS = 12         # onaylanmayan öneri süresi dolar
    USE_AI_SCORE = True             # ANTHROPIC_API_KEY varsa AI düzeltmesi (-15..+15)
    AI_MODEL = "claude-sonnet-4-6"

    # sembol: (varlık_sınıfı, yfinance/TEFAS kodu)
    UNIVERSE = {
        "BTC":   ("kripto", "BTC-USD"),
        "ETH":   ("kripto", "ETH-USD"),
        "SOL":   ("kripto", "SOL-USD"),
        "THYAO": ("bist",   "THYAO.IS"),
        "ASELS": ("bist",   "ASELS.IS"),
        "TUPRS": ("bist",   "TUPRS.IS"),
        "KCHOL": ("bist",   "KCHOL.IS"),
        "BIMAS": ("bist",   "BIMAS.IS"),
        "GRAM_ALTIN": ("altin", "GC=F"),
        "TCD": ("tefas", "TCD"),   # örnek fon kodları; kendi fonlarınla değiştir
        "MAC": ("tefas", "MAC"),
    }

# ===== data =====


def _yf(sym):
    df = yf.download(sym, period="9mo", interval="1d", progress=False, auto_adjust=True)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna().astype(float)


def _tefas(code):
    from tefas import Crawler
    end = date.today()
    start = end - timedelta(days=200)
    df = Crawler().fetch(start=start.isoformat(), end=end.isoformat(),
                         name=code, columns=["date", "price"])
    s = pd.Series(df["price"].astype(float).values, index=pd.to_datetime(df["date"]))
    return s.sort_index().dropna()


def load_all():
    """{sembol: (sınıf, TL cinsinden kapanış serisi)} döndürür; hatalı olanları atlar."""
    usd = _yf("USDTRY=X")
    out = {}
    for name, (asset, code) in C.UNIVERSE.items():
        try:
            if asset == "tefas":
                s = _tefas(code)
            else:
                s = _yf(code)
                if asset in ("kripto", "altin"):
                    fx = usd.reindex(s.index).ffill().bfill()
                    s = s * fx
                    if asset == "altin":
                        s = s / 31.1035          # ons -> gram
            if len(s) >= 55:
                out[name] = (asset, s)
        except Exception as e:  # tek varlık hatası tüm sistemi durdurmasın
            print(f"[uyarı] {name} alınamadı: {e}")
    return out

# ===== scoring =====


def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    v = (100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1]
    return 50.0 if pd.isna(v) else float(v)


def tech_score(s: pd.Series):
    p = s.iloc[-1]
    sma20, sma50 = s.rolling(20).mean().iloc[-1], s.rolling(50).mean().iloc[-1]
    trend = 30 if p > sma20 > sma50 else 18 if p > sma50 else 8 if p > sma20 else 0
    r = _rsi(s)
    rsi_pts = 25 if 45 <= r <= 65 else 15 if (35 <= r < 45 or 65 < r <= 72) else 5 if r < 35 else 0
    mom = float(s.iloc[-1] / s.iloc[-21] - 1)
    mom_pts = float(np.clip(mom / 0.15, 0, 1) * 25)
    vol = float(s.pct_change().tail(30).std() * np.sqrt(252))
    vol_pts = float(np.clip(1 - (vol - 0.2) / 0.8, 0, 1) * 20)
    score = trend + rsi_pts + mom_pts + vol_pts
    return score, {"rsi": round(r, 1), "mom20": round(mom * 100, 1),
                   "vol": round(vol * 100, 1), "trend": trend}


def ai_adjust(name, asset, metrics):
    """Claude'dan -15..+15 arası düzeltme ve kısa gerekçe. Anahtar yoksa (0, '')."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not (C.USE_AI_SCORE and key):
        return 0, ""
    prompt = (f"Varlık: {name} ({asset}). Teknik göstergeler: {json.dumps(metrics)}. "
              "Bu verilere bakarak kısa vadeli risk/getiri dengesini değerlendir. "
              'Sadece JSON dön: {"adj": -15 ile 15 arası tam sayı, "note": "en fazla 15 kelime"}')
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=30,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": C.AI_MODEL, "max_tokens": 200,
                                "messages": [{"role": "user", "content": prompt}]})
        txt = r.json()["content"][0]["text"].replace("```json", "").replace("```", "").strip()
        j = json.loads(txt)
        return int(np.clip(j["adj"], -15, 15)), str(j.get("note", ""))
    except Exception as e:
        print(f"[uyarı] AI skoru alınamadı: {e}")
        return 0, ""


def total_score(name, asset, s, use_ai=True):
    base, m = tech_score(s)
    adj, note = ai_adjust(name, asset, m) if use_ai else (0, "")
    return float(np.clip(base + adj, 0, 100)), m, note

# ===== portfolio =====

STATE_FILE = "state.json"


def load():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"cash": C.INITIAL_CASH, "positions": {}, "pending": {},
            "last_update_id": 0, "history": []}


def save(pf):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)


def equity(pf, prices):
    return pf["cash"] + sum(p["qty"] * prices.get(sym, p["avg_cost"])
                            for sym, p in pf["positions"].items())


def max_spend(pf, prices):
    """Nakit rezervini koruyarak harcanabilecek en fazla tutar."""
    eq = equity(pf, prices)
    return max(0.0, pf["cash"] - C.CASH_RESERVE_PCT * eq)


def buy(pf, sym, asset, amount, price, prices):
    amount = min(amount, max_spend(pf, prices))
    if amount < C.MIN_TRADE_TL:
        return None
    qty = amount / price
    pos = pf["positions"].get(sym)
    if pos:
        tot = pos["qty"] + qty
        pos["avg_cost"] = (pos["avg_cost"] * pos["qty"] + amount) / tot
        pos["qty"], pos["peak"] = tot, max(pos["peak"], price)
    else:
        pf["positions"][sym] = {"asset": asset, "qty": qty, "avg_cost": price, "peak": price}
    pf["cash"] -= amount
    _log(pf, "AL", sym, qty, price)
    return amount


def sell_all(pf, sym, price):
    pos = pf["positions"].pop(sym)
    pf["cash"] += pos["qty"] * price
    pnl = (price / pos["avg_cost"] - 1) * 100
    _log(pf, "SAT", sym, pos["qty"], price)
    return pnl


def _log(pf, side, sym, qty, price):
    pf["history"].append({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "side": side, "sym": sym, "qty": qty, "price": price})
    pf["history"] = pf["history"][-500:]

# ===== telegram =====

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID", ""))
API = f"https://api.telegram.org/bot{TOKEN}"


def _call(method, **payload):
    r = requests.post(f"{API}/{method}", json=payload, timeout=30)
    return r.json()


def send(text, buttons=None):
    p = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    if buttons:
        p["reply_markup"] = {"inline_keyboard": [buttons]}
    return _call("sendMessage", **p)


def proposal_buttons(pid):
    return [{"text": "✅ Onayla", "callback_data": f"ok:{pid}"},
            {"text": "❌ Reddet", "callback_data": f"no:{pid}"}]


def get_updates(offset):
    return _call("getUpdates", offset=offset, timeout=0,
                 allowed_updates=["callback_query"]).get("result", [])


def answer(cb_id, text):
    _call("answerCallbackQuery", callback_query_id=cb_id, text=text)


def edit(msg, text):
    _call("editMessageText", chat_id=msg["chat"]["id"], message_id=msg["message_id"],
          text=text, parse_mode="HTML")

P = types.SimpleNamespace(load=load, save=save, equity=equity, max_spend=max_spend, buy=buy, sell_all=sell_all)
tg = types.SimpleNamespace(send=send, proposal_buttons=proposal_buttons, get_updates=get_updates,
                           answer=answer, edit=edit, CHAT_ID=CHAT_ID)

# ===== main =====

TR = ZoneInfo("Europe/Istanbul")
tl = lambda x: f"{x:,.0f} ₺".replace(",", ".")


def market_open(asset, now):
    if asset == "kripto":
        return True
    if now.weekday() >= 5:
        return False
    if asset == "bist":
        return 10 <= now.hour < 18
    return True  # altın / TEFAS: hafta içi günlük veri


def process_approvals(pf, prices):
    for u in tg.get_updates(pf["last_update_id"] + 1):
        pf["last_update_id"] = u["update_id"]
        cb = u.get("callback_query")
        if not cb or str(cb["message"]["chat"]["id"]) != tg.CHAT_ID:
            continue  # sadece senin sohbetin
        action, pid = cb["data"].split(":", 1)
        prop = pf["pending"].pop(pid, None)
        if not prop:
            tg.answer(cb["id"], "Öneri bulunamadı / süresi dolmuş")
            continue
        if action == "no":
            tg.answer(cb["id"], "Reddedildi")
            tg.edit(cb["message"], f"❌ <b>{prop['sym']}</b> alımı reddedildi.")
            continue
        price = prices.get(prop["sym"])
        spent = P.buy(pf, prop["sym"], prop["asset"], prop["amount"], price, prices) if price else None
        if spent:
            tg.answer(cb["id"], "Alındı")
            tg.edit(cb["message"], f"✅ <b>{prop['sym']}</b> alındı: {tl(spent)} @ {price:,.2f}")
        else:
            tg.answer(cb["id"], "Nakit rezervi nedeniyle alınamadı")
            tg.edit(cb["message"], f"⚠️ <b>{prop['sym']}</b> alınamadı (nakit rezervi / fiyat yok).")


def expire_pending(pf, now):
    for pid, p in list(pf["pending"].items()):
        if now - datetime.fromisoformat(p["created"]) > timedelta(hours=C.PROPOSAL_TTL_HOURS):
            pf["pending"].pop(pid)


def auto_sells(pf, data, prices, now):
    for sym, pos in list(pf["positions"].items()):
        if sym not in prices or not market_open(pos["asset"], now):
            continue
        price = prices[sym]
        pos["peak"] = max(pos["peak"], price)
        pnl = price / pos["avg_cost"] - 1
        drop = price / pos["peak"] - 1
        reason = None
        if pnl <= C.STOP_LOSS:
            reason = f"Zarar kes ({pnl:.1%})"
        elif pnl >= C.TAKE_PROFIT:
            reason = f"Kâr al ({pnl:.1%})"
        elif pnl > 0 and drop <= -C.TRAILING_STOP:
            reason = f"İzleyen stop (zirveden {drop:.1%})"
        else:
            score, _, _ = total_score(sym, pos["asset"], data[sym][1], use_ai=False)
            if score < C.SELL_SCORE_BELOW:
                reason = f"Skor düştü ({score:.0f})"
        if reason:
            res = P.sell_all(pf, sym, price)
            tg.send(f"🔴 <b>SATIŞ: {sym}</b>\n{reason}\nGetiri: {res:+.1f}% @ {price:,.2f}")


def propose_buys(pf, data, prices, now):
    pending_syms = {p["sym"] for p in pf["pending"].values()}
    cands = []
    for sym, (asset, s) in data.items():
        if sym in pf["positions"] or sym in pending_syms or not market_open(asset, now):
            continue
        base, _, _ = total_score(sym, asset, s, use_ai=False)  # ucuz ön eleme
        if base >= C.BUY_SCORE_THRESHOLD - 15:
            cands.append((base, sym, asset))
    cands.sort(reverse=True)

    sent = 0
    for _, sym, asset in cands:
        if sent >= C.MAX_PROPOSALS_PER_RUN:
            break
        score, m, note = total_score(sym, asset, data[sym][1])  # AI dahil nihai skor
        if score < C.BUY_SCORE_THRESHOLD:
            continue
        eq = P.equity(pf, prices)
        reserved = sum(p["amount"] for p in pf["pending"].values())
        room = P.max_spend(pf, prices) - reserved
        weight = 0.5 + 0.5 * min(1, (score - C.BUY_SCORE_THRESHOLD) / 30)
        amount = min(C.MAX_POSITION_PCT * eq * weight, room)
        if amount < C.MIN_TRADE_TL:
            continue
        pid = uuid.uuid4().hex[:8]
        pf["pending"][pid] = {"sym": sym, "asset": asset, "amount": amount,
                              "created": now.replace(tzinfo=None).isoformat()}
        txt = (f"🟢 <b>ALIM FIRSATI: {sym}</b> ({asset})\n"
               f"Skor: <b>{score:.0f}/100</b>\nFiyat: {prices[sym]:,.2f} ₺\n"
               f"Önerilen tutar: {tl(amount)}\n"
               f"RSI {m['rsi']} | 20g {m['mom20']:+.1f}% | Vol {m['vol']}%")
        if note:
            txt += f"\n🤖 {note}"
        tg.send(txt, tg.proposal_buttons(pid))
        sent += 1


def summary(pf, prices):
    eq = P.equity(pf, prices)
    lines = [f"📊 <b>Günlük özet</b>\nToplam: {tl(eq)} ({eq / C.INITIAL_CASH - 1:+.1%})\n"
             f"Nakit: {tl(pf['cash'])} ({pf['cash'] / eq:.0%})"]
    for sym, p in pf["positions"].items():
        pr = prices.get(sym, p["avg_cost"])
        lines.append(f"• {sym}: {(pr / p['avg_cost'] - 1):+.1%} ({tl(p['qty'] * pr)})")
    tg.send("\n".join(lines))


def main():
    now = datetime.now(TR)
    if not os.path.exists(STATE_FILE):
        tg.send("✅ Portföy botu bağlandı. Fırsat yakalayınca butonlu mesaj göndereceğim.")
    pf = P.load()
    data = load_all()
    prices = {s: float(v[1].iloc[-1]) for s, v in data.items()}

    process_approvals(pf, prices)
    expire_pending(pf, now.replace(tzinfo=None))
    auto_sells(pf, data, prices, now)
    propose_buys(pf, data, prices, now)
    if now.weekday() < 5 and now.hour == 18 and now.minute < 30:
        summary(pf, prices)
    P.save(pf)


if __name__ == "__main__":
    main()
