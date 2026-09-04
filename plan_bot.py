import os
import time
import json
import datetime
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OKX = "https://www.okx.com"
TGAPI = "https://api.telegram.org"
STATE_FILE = "plan_state.json"


def _g(k, d):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
ACCOUNT = float(_g("ACCOUNT_USDT", "10000"))
MIN_VOL = float(_g("MIN_24H_VOLUME_USDT", "5000000"))
MIN_RR = float(_g("MIN_RR", "1.5"))
WORKERS = int(_g("WORKERS", "4"))
SCALP_ON = _g("SCALP_ENABLED", "true").lower() == "true"
SEND_EMPTY = _g("SEND_WHEN_EMPTY", "false").lower() == "true"
EXCLUDE = set(s.strip().upper() for s in _g("EXCLUDE_BASES", "").split(",") if s.strip())

TIERS = {
    "swing": dict(name="① 코어 스윙", risk=1.0, cap=4.0, maxpos=2, cool=24),
    "mid":   dict(name="② 중단타",    risk=0.7, cap=2.5, maxpos=3, cool=6),
    "scalp": dict(name="③ 스캘핑",    risk=0.4, cap=1.5, maxpos=1, cool=2),
}
JP = {"1Dutc": (2.5, 3.0, 12.0), "1H": (3.0, 1.2, 4.0), "30m": (2.3, 0.9, 4.0), "15m": (2.0, 0.7, 4.0)}

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 plan-bot"})


def okx_get(path, params, tries=5):
    for i in range(tries):
        try:
            r = S.get(OKX + path, params=params, timeout=25)
        except requests.RequestException:
            time.sleep(1 + i)
            continue
        if r.status_code == 200:
            j = r.json()
            return (j.get("data") or []) if j.get("code") == "0" else None
        if r.status_code in (403, 429):
            time.sleep(1.0 + i * 1.5)
            continue
        time.sleep(1 + i)
    return None


def sma(a, n):
    a = np.asarray(a, float)
    out = np.full(a.shape, np.nan)
    if len(a) >= n:
        c = np.cumsum(np.insert(a, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rma(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def rsi(c, n=14):
    d = np.diff(c)
    ru = rma(np.where(d > 0, d, 0.0), n)
    rd = rma(np.where(d < 0, -d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + ru / rd)
    out = np.where(rd == 0, 100.0, out)
    return np.concatenate([[np.nan], out])


def atr(h, l, c, n=14):
    pc = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return rma(tr, n)


def cci(src, n=20):
    src = np.asarray(src, float)
    ma = sma(src, n)
    md = np.full(src.shape, np.nan)
    for i in range(n - 1, len(src)):
        w = src[i - n + 1:i + 1]
        md[i] = np.mean(np.abs(w - ma[i]))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (src - ma) / (0.015 * md)
    out[~np.isfinite(out)] = np.nan
    return out


def crossed_up(s, lv):
    s = s[np.isfinite(s)]
    return bool(len(s) >= 2 and s[-2] < lv and s[-1] >= lv)


def klines(instid, bar, limit=250):
    data = okx_get("/api/v5/market/candles",
                   {"instId": instid, "bar": bar, "limit": str(min(limit, 300))})
    if not data or len(data) < 60:
        return None
    rows = [x for x in data if len(x) > 8 and x[8] == "1"]
    if len(rows) < 60:
        rows = data[1:]
    rows = rows[::-1]
    return (np.array([float(x[2]) for x in rows]), np.array([float(x[3]) for x in rows]),
            np.array([float(x[4]) for x in rows]), np.array([float(x[5]) for x in rows]))


def swings(h, l, c, r, a, am, pf, pc):
    P, T, R = [], [], []
    d, ext = 1, c[0]
    er = r[0] if np.isfinite(r[0]) else 50.0
    for i in range(1, len(c)):
        px = c[i]
        av = a[i] if np.isfinite(a[i]) else px * pf / 100.0
        thr = min(max(av * am, px * pf / 100.0), px * pc / 100.0)
        ri = r[i] if np.isfinite(r[i]) else er
        if d == 1:
            if h[i] > ext:
                ext, er = h[i], ri
            elif c[i] < ext - thr:
                P.append(ext), T.append(1), R.append(er)
                d, ext, er = -1, l[i], ri
        else:
            if l[i] < ext:
                ext, er = l[i], ri
            elif c[i] > ext + thr:
                P.append(ext), T.append(-1), R.append(er)
                d, ext, er = 1, h[i], ri
    return P, T, R


def nth(P, T, R, ty, k):
    cnt = 0
    for i in range(len(T) - 1, -1, -1):
        if T[i] == ty:
            if cnt == k:
                return P[i], R[i]
            cnt += 1
    return None, None


def jade(hlc, params):
    if hlc is None:
        return None
    h, l, c, _ = hlc
    if len(c) < 60:
        return None
    r = rsi(c)
    a = atr(h, l, c)
    P, T, R = swings(h, l, c, r, a, *params)
    S1, S1r = nth(P, T, R, 1, 0)
    S2, S2r = nth(P, T, R, 1, 1)
    L1, L1r = nth(P, T, R, -1, 0)
    L2, L2r = nth(P, T, R, -1, 1)
    px = float(c[-1])
    rr = float(r[-1]) if np.isfinite(r[-1]) else 50.0
    up = L1 is not None and L2 is not None and L1 > L2 and px > L1
    dn = S1 is not None and S2 is not None and S1 < S2 and px < S1
    bull2 = None not in (L1, L2, L1r, L2r) and L1 < L2 and L1r > L2r
    bear2 = None not in (S1, S2, S1r, S2r) and S1 > S2 and S1r < S2r
    uL = L2 if (L1 is not None and px < L1 and L2 is not None) else L1
    uH = S2 if (S1 is not None and px > S1 and S2 is not None) else S1
    fR = (uH - uL) if (uH is not None and uL is not None and uH > uL) else None
    rl = (uH - px) / fR if fR else None
    rs = (px - uL) / fR if fR else None
    mid = 43.0 < rr < 57.0
    return dict(px=px, rsi=rr, up=up, dn=dn, prev_low=L1, prev_high=S1,
                retrL=rl, retrS=rs,
                f618=(uH - 0.618 * fR) if fR else None,
                f786=(uH - 0.786 * fR) if fR else None,
                s618=(uL + 0.618 * fR) if fR else None,
                s786=(uL + 0.786 * fR) if fR else None,
                longEntry=bool(up and rl is not None and 0.618 <= rl <= 0.786 and bull2 and not mid),
                shortEntry=bool(dn and rs is not None and 0.618 <= rs <= 0.786 and bear2 and not mid))


def stage1(instid):
    k = klines(instid, "1Dutc", 250)
    if k is None:
        return None
    h, l, c, _ = k
    s22 = sma(c, 22)
    b20 = sma(c, 20)
    if not np.isfinite(s22[-1]):
        return None
    j = jade(k, JP["1Dutc"])
    return dict(close=float(c[-1]), above=bool(c[-1] > s22[-1]), sma22=float(s22[-1]),
                bbmid=float(b20[-1]) if np.isfinite(b20[-1]) else None,
                cci_d=crossed_up(cci(c), -100),
                prev_high=(j or {}).get("prev_high"))


def stage2(instid):
    out = dict(c12=False, c4=False, volx=None, bb4=None)
    k = klines(instid, "12Hutc", 250)
    if k is not None:
        out["c12"] = crossed_up(cci(k[2]), -80)
    k = klines(instid, "4H", 250)
    if k is not None:
        c4, v4 = k[2], k[3]
        vm = sma(v4, 20)
        if np.isfinite(vm[-1]) and vm[-1] > 0:
            out["volx"] = float(v4[-1] / vm[-1])
        b = sma(c4, 20)
        out["bb4"] = float(b[-1]) if np.isfinite(b[-1]) else None
        out["c4"] = crossed_up(cci(c4), -100) and bool(out["volx"] and out["volx"] >= 1.5)
    return out


def stage3(instid):
    return {tf: jade(klines(instid, tf, 250), JP[tf]) for tf in ("1H", "30m", "15m")}


def mk(instid, tier, side, ref, zone_lo, zone_hi, raw_stop, tp1, tp2, why, tf_tag):
    t = TIERS[tier]
    if ref is None or raw_stop is None or tp1 is None:
        return None
    if side == "long":
        stop = max(raw_stop, ref * (1 - t["cap"] / 100.0))
        if stop >= ref or tp1 <= ref:
            return None
        risk_d, rew_d = ref - stop, tp1 - ref
    else:
        stop = min(raw_stop, ref * (1 + t["cap"] / 100.0))
        if stop <= ref or tp1 >= ref:
            return None
        risk_d, rew_d = stop - ref, ref - tp1
    rr = rew_d / risk_d
    if rr < MIN_RR:
        return None
    stop_pct = risk_d / ref * 100.0
    pos_usd = (ACCOUNT * t["risk"] / 100.0) / (stop_pct / 100.0)
    return dict(instid=instid, tier=tier, side=side, ref=ref, stop=stop, stop_pct=stop_pct,
                zlo=zone_lo, zhi=zone_hi, tp1=tp1, tp2=tp2, rr=rr,
                pos=pos_usd, qty=pos_usd / ref, why=why, tf=tf_tag)


def build(instid, s1, s2, j, scalp_ok):
    out = []
    if s1["above"] and (s1["cci_d"] or s2["c12"]):
        a = j.get("1H")
        if a and a["longEntry"]:
            src = "일봉 CCI -100 상향" if s1["cci_d"] else "12h CCI -80 상향"
            p = mk(instid, "swing", "long", a["px"], a["f786"], a["f618"], a["prev_low"],
                   s1["bbmid"], s1["prev_high"],
                   f"4BC {src} · 일봉 SMA22 위  /  제이드 1H 롱진입(되돌림 {a['retrL']*100:.0f}%)", "1H")
            if p:
                out.append(p)
    if s1["above"] and s2["c4"]:
        h1 = j.get("1H")
        for tf in ("30m", "15m"):
            a = j.get(tf)
            if h1 and h1["up"] and a and a["longEntry"]:
                p = mk(instid, "mid", "long", a["px"], a["f786"], a["f618"], a["prev_low"],
                       s2["bb4"], (h1 or {}).get("prev_high"),
                       f"4BC 4H CCI -100 상향 · 거래량 {s2['volx']:.1f}x · 일봉 SMA22 위"
                       f"  /  제이드 1H 상승파동 + {tf} 롱진입(되돌림 {a['retrL']*100:.0f}%)", tf)
                if p:
                    out.append(p)
                    break
    if scalp_ok:
        h1, h30, s15 = j.get("1H"), j.get("30m"), j.get("15m")
        if h1 and h30 and s15:
            if s1["above"] and h1["up"] and h30["up"] and s15["longEntry"]:
                r1 = s15["px"] - max(s15["prev_low"] or 0, s15["px"] * 0.985)
                p = mk(instid, "scalp", "long", s15["px"], s15["f786"], s15["f618"], s15["prev_low"],
                       s15["px"] + r1, s15["px"] + 1.5 * r1,
                       "제이드 15m A급(3TF 정렬) · 일봉 SMA22 위", "15m")
                if p:
                    out.append(p)
            elif (not s1["above"]) and h1["dn"] and h30["dn"] and s15["shortEntry"]:
                r1 = min(s15["prev_high"] or 1e18, s15["px"] * 1.015) - s15["px"]
                p = mk(instid, "scalp", "short", s15["px"], s15["s618"], s15["s786"], s15["prev_high"],
                       s15["px"] - r1, s15["px"] - 1.5 * r1,
                       "제이드 15m A급(3TF 정렬) · 일봉 SMA22 아래", "15m")
                if p:
                    out.append(p)
    return out


def fp(p):
    if p is None:
        return "-"
    if p >= 100:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.3f}"
    return f"{p:.8f}".rstrip("0").rstrip(".")


def sym(i):
    return i.split("-")[0] + "USDT"


def card(p):
    t = TIERS[p["tier"]]
    e = "\U0001F7E2" if p["side"] == "long" else "\U0001F534"
    sd = "롱" if p["side"] == "long" else "숏"
    zs = sorted([x for x in (p["zlo"], p["zhi"]) if x is not None])
    lo = zs[0] if zs else None
    hi = zs[-1] if zs else None
    L = [f"{e} <b>[{t['name']}]</b> {sym(p['instid'])}  현재 {fp(p['ref'])}  <i>{sd}</i>",
         f"  {p['why']}",
         "  ───────────────",
         f"  진입   {fp(lo)} ~ {fp(hi)}   <i>({p['tf']} 피보 0.618~0.786)</i>",
         f"  손절   {fp(p['stop'])}   <i>({p['stop_pct']:.1f}%)</i>",
         f"  익절1  {fp(p['tp1'])}   <i>50% 청산 → 손절 본절로</i>",
         f"  익절2  {fp(p['tp2'])}",
         f"  손익비 1 : {p['rr']:.1f}",
         f"  수량   계좌 ${ACCOUNT:,.0f} · 리스크 {t['risk']}% → <b>${p['pos']:,.0f}</b> ({p['qty']:,.4f})"]
    return "\n".join(L)


def send(txt):
    r = S.post(f"{TGAPI}/bot{TOKEN}/sendMessage",
               data={"chat_id": CHAT, "text": txt, "parse_mode": "HTML",
                     "disable_web_page_preview": "true"}, timeout=25)
    if r.status_code != 200:
        print("telegram error:", r.status_code, r.text[:300])


def split(msg, lim=3800):
    if len(msg) <= lim:
        return [msg]
    parts, cur = [], ""
    for b in msg.split("\n\n"):
        if len(cur) + len(b) + 2 > lim and cur:
            parts.append(cur)
            cur = ""
        cur += b + "\n\n"
    if cur:
        parts.append(cur)
    return parts


def pdt(s):
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        return None


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    nk = datetime.datetime.now(kst)
    scalp_ok = SCALP_ON and (nk.weekday() >= 5 or 19 <= nk.hour <= 23)

    inst = okx_get("/api/v5/public/instruments", {"instType": "SWAP"})
    if not inst:
        print("instruments fetch failed")
        return
    syms = [it["instId"] for it in inst
            if it.get("state") == "live" and it.get("settleCcy") == "USDT"
            and it.get("instId", "").endswith("-USDT-SWAP")
            and it["instId"].split("-")[0].upper() not in EXCLUDE]
    tk = okx_get("/api/v5/market/tickers", {"instType": "SWAP"}) or []
    vol = {}
    for t in tk:
        try:
            vol[t["instId"]] = float(t.get("volCcy24h") or 0) * float(t.get("last") or 0)
        except (TypeError, ValueError):
            vol[t["instId"]] = 0.0
    syms = [s for s in syms if vol.get(s, 0) >= MIN_VOL]
    syms.sort(key=lambda s: -vol.get(s, 0))
    print(f"[1] universe {len(syms)}  scalp_window={scalp_ok}")

    s1m = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fu = {ex.submit(stage1, s): s for s in syms}
        for f in as_completed(fu):
            try:
                r = f.result()
            except Exception:
                r = None
            if r:
                s1m[fu[f]] = r
    cand = [s for s in syms if s in s1m and (s1m[s]["above"] or scalp_ok)]
    print(f"[2] after daily screen {len(cand)}")

    s2m = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fu = {ex.submit(stage2, s): s for s in cand}
        for f in as_completed(fu):
            try:
                s2m[fu[f]] = f.result()
            except Exception:
                pass
    hot = [s for s in cand if s in s2m and (s1m[s]["cci_d"] or s2m[s]["c12"] or s2m[s]["c4"])]
    if scalp_ok:
        hot = list(dict.fromkeys(hot + cand[:30]))
    print(f"[3] 4BC hits + scalp pool {len(hot)}")

    plans = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fu = {ex.submit(stage3, s): s for s in hot}
        for f in as_completed(fu):
            s = fu[f]
            try:
                j = f.result()
            except Exception:
                continue
            plans += build(s, s1m[s], s2m.get(s, dict(c12=False, c4=False, volx=None, bb4=None)),
                           j, scalp_ok)
    print(f"[4] plans passing RR>={MIN_RR} : {len(plans)}")

    now = datetime.datetime.now(datetime.timezone.utc)
    st = {k: v for k, v in (json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}).items()
          if pdt(v) and pdt(v) > now - datetime.timedelta(hours=72)}
    fresh = []
    for p in plans:
        key = f"{sym(p['instid'])}:{p['tier']}:{p['side']}"
        prev = pdt(st.get(key))
        if prev and (now - prev).total_seconds() < TIERS[p["tier"]]["cool"] * 3600:
            continue
        fresh.append(p)
        st[key] = now.isoformat()
    json.dump(st, open(STATE_FILE, "w"))

    stamp = nk.strftime("%Y-%m-%d %H:%M")
    if not fresh:
        print("no fresh plans")
        if SEND_EMPTY:
            send(f"\U0001F4CB <b>매매 플랜</b> · {stamp} KST · 신호 없음")
        return

    rank = {"swing": 0, "mid": 1, "scalp": 2}
    fresh.sort(key=lambda p: (rank[p["tier"]], -p["rr"]))
    head = (f"\U0001F4CB <b>매매 플랜</b> · {stamp} KST · {len(fresh)}건\n"
            f"<i>동시 보유 ① 2 / ② 3 / ③ 1 · 총 리스크 4% 상한 · "
            f"진입과 동시에 손절 주문 등록</i>")
    for ch in split(head + "\n\n" + "\n\n".join(card(p) for p in fresh)):
        send(ch)
        time.sleep(0.4)


if __name__ == "__main__":
    main()
