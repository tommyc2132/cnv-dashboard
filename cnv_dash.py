# cnv_dash.py
import os
import calendar
from datetime import date

import pandas as pd
import streamlit as st

from google.cloud import bigquery
from google.oauth2 import service_account

# =========================================================
# 0) 고정 설정
# =========================================================
PROJECT_ID = os.environ.get("PROJECT_ID", "strange-reducer-474905-g1").strip()

DEFAULT_TABLE_FQN = f"{PROJECT_ID}.streamlit.cnv_dash_tbl"
SOURCE_FQN = os.environ.get("CNV_SOURCE", DEFAULT_TABLE_FQN).strip()

GOOGLE_KEY_PATH = os.environ.get(
    "GOOGLE_KEY_PATH",
    r"C:\tommy\BigQuery\strange-reducer-474905-g1-946a9f4f9fac.json"
).strip()

BQ_LOCATION = os.environ.get("BQ_LOCATION", "asia-northeast3").strip()

# =========================================================
# 1) Streamlit 기본
# =========================================================
st.set_page_config(page_title="상담 → 주문(0~48h) 대시보드", layout="wide")
st.title("📊 상담 → 주문전환 측정 (0~48h) 대시보드 ")

# 🔥 전환율 정의 노티
st.markdown(
    """
<div style="
  background-color:#fff4e5;
  border-left:6px solid #ff9800;
  padding:12px 14px;
  border-radius:6px;
  font-size:0.95rem;
  line-height:1.5;
">
<b>전환율 산정 기준 변경</b><br/>
- 본 대시보드는 <b>**무효 상담을 모수에서 제외**</b>한 후 전환율을 계산합니다.<br/>
- 정상 상담 기준의 <b>실질 구매 전환 성과</b>를 보기 위함.<br/>
- 전환 조건: <b>상담 후 48시간 이내</b>, <b>C주문 제외</b>, <b>결제금액 &gt; 0</b>
</div>
""",
    unsafe_allow_html=True
)

st.caption(" · 날짜 기준: 상담일자(inbound_date)")

# =========================================================
# 2) BigQuery Client (서울 리전 고정)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_bq_client():
    # 1) 로컬 키파일 우선
    if GOOGLE_KEY_PATH and os.path.exists(GOOGLE_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(
            project=PROJECT_ID,
            credentials=creds,
            location=BQ_LOCATION
        )

    # 2) Secrets
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return bigquery.Client(
                project=PROJECT_ID,
                credentials=creds,
                location=BQ_LOCATION
            )
    except Exception:
        pass

    # 3) ADC
    return bigquery.Client(
        project=PROJECT_ID,
        location=BQ_LOCATION
    )

# =========================================================
# 3) UI 한글 컬럼 매핑
# =========================================================
KOR_COL_MAP = {
    "inbound_date": "상담일자",
    "inbound_ts": "상담시점",
    "inbound_channel": "인입채널",
    "ticket_id": "티켓번호",
    "agent_center": "센터명",
    "agent_name": "담당자",
    "brand_name": "상담브랜드",
    "matched_brand": "주문브랜드",
    "category_lv1": "문의유형_대",
    "category_lv2": "문의유형_중",
    "category_lv3": "문의유형_소",
    "customer_phone": "고객휴대폰",
    "converted_yn": "전환여부",
    "first_order_ts": "주문시점",
    "order_cnt": "전환주문수",
    "order_amount": "주문금액",
    "min_leadtime_h": "리드타임",
    "order_nos": "전환주문번호",
    "sellers": "판매처",
    "matched_by": "매칭기준",
    "ticket_cnt": "티켓수",
    "conv_rate": "전환율",
}

def apply_kor_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in KOR_COL_MAP.items() if k in df.columns})

# =========================================================
# 3-1) 🔥 Rank: 화면은 "얇게" 보이게 (index로) / CSV는 컬럼으로
# =========================================================
def with_rank_index(df: pd.DataFrame, index_name: str = "Rank") -> pd.DataFrame:
    out = df.copy()
    out.index = range(1, len(out) + 1)
    out.index.name = index_name
    return out

def with_rank_col(df: pd.DataFrame, col_name: str = "Rank") -> pd.DataFrame:
    out = df.copy()
    out.insert(0, col_name, range(1, len(out) + 1))
    return out

# =========================================================
# 4) 비용 캡
# =========================================================
def bytes_from_gb(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)

# =========================================================
# 5) 기간 선택: 월 드롭다운(2026-01부터) + 캘린더(자유 수정)
# =========================================================
START_MONTH = date(2026, 1, 1)

def month_start_end(y: int, m: int):
    s = date(y, m, 1)
    e = date(y, m, calendar.monthrange(y, m)[1])
    return s, e

def build_month_options(start_month: date, end_month: date):
    start_idx = start_month.year * 12 + start_month.month
    end_idx = end_month.year * 12 + end_month.month
    opts = []
    for idx in range(end_idx, start_idx - 1, -1):
        y = (idx - 1) // 12
        m = (idx - 1) % 12 + 1
        opts.append(f"{y}-{m:02d}")
    return opts

today = date.today()
this_month_start = date(today.year, today.month, 1)
month_options = build_month_options(START_MONTH, this_month_start)

def on_month_change():
    sel = st.session_state["selected_month"]
    y, m = map(int, sel.split("-"))
    s, e = month_start_end(y, m)
    st.session_state["date_from"] = s
    st.session_state["date_to"] = e

if "selected_month" not in st.session_state:
    st.session_state["selected_month"] = f"{today.year}-{today.month:02d}"
if st.session_state["selected_month"] not in month_options:
    st.session_state["selected_month"] = month_options[0]

if "date_from" not in st.session_state or "date_to" not in st.session_state:
    y, m = map(int, st.session_state["selected_month"].split("-"))
    s, e = month_start_end(y, m)
    st.session_state["date_from"] = s
    st.session_state["date_to"] = e

st.sidebar.header("기간 선택")
st.sidebar.selectbox(
    "월 선택",
    options=month_options,
    key="selected_month",
    on_change=on_month_change,
)

date_from = st.sidebar.date_input("시작일", key="date_from")
date_to = st.sidebar.date_input("종료일", key="date_to")

if date_from > date_to:
    st.sidebar.error("시작일이 종료일보다 클 수 없습니다.")
    st.stop()

st.sidebar.markdown(
    """
<div style="
  background-color:#e7f1ff;
  border-left:6px solid #1f6feb;
  padding:10px 12px;
  border-radius:6px;
  font-size:0.9rem;
  line-height:1.4;
">
  ⭐날짜 Default 값 : 이번달<br/>
  ⭐전환율 정의 = SUM(전환주문수) / 티켓수<br/><br/>
  <b>작업자: 조영우</b>
</div>
""",
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.header("피벗 설정")

available_dims = [
    "agent_name",
    "brand_name",
    "category_lv1", "category_lv2", "category_lv3",
    "matched_brand",
    "converted_yn",
    "matched_by",
]

rows = st.sidebar.multiselect("ROWS (드릴다운)", options=available_dims, default=["agent_name"])

col_candidates = [c for c in available_dims if c not in set(rows)]
col = st.sidebar.selectbox("COLUMNS (선택)", options=["(없음)"] + col_candidates, index=0)
col = None if col == "(없음)" else col

min_ticket = st.sidebar.number_input("최소 티켓수(필터, COLUMNS 없을 때)", min_value=0, value=0, step=10)
sort_key = st.sidebar.selectbox(
    "정렬",
    options=["ticket_cnt", "order_cnt", "order_amount", "conv_rate"],
    index=3
)
sort_desc = st.sidebar.checkbox("내림차순", value=True)

st.sidebar.divider()
st.sidebar.header("BigQuery 비용 안전장치")
max_gb = st.sidebar.slider("최대 스캔 허용(GB)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
max_bytes_billed = bytes_from_gb(max_gb)

raw_limit = st.sidebar.selectbox(
    "로우데이터 기본 LIMIT",
    options=[1000, 5000, 20000, 50000, 100000],
    index=3
)

# =========================================================
# 6) 집계 로드
# =========================================================
@st.cache_data(ttl=300, show_spinner=True)
def load_agg(date_from, date_to, rows, col, max_bytes_billed: int) -> pd.DataFrame:
    client = get_bq_client()

    if not rows:
        raise ValueError("ROWS는 최소 1개 필요")

    if col is not None and col in rows:
        col = None

    dim_fields = ["agent_center"] + rows + ([col] if col else [])
    dim_select = ",\n      ".join(dim_fields)
    dim_groupby = ", ".join(dim_fields)

    sql = f"""
    SELECT
      {dim_select},
      COUNT(1) AS ticket_cnt,
      SUM(CAST(order_cnt AS INT64)) AS order_cnt,
      SUM(CAST(order_amount AS INT64)) AS order_amount
    FROM `{SOURCE_FQN}`
    WHERE inbound_date >= @date_from
      AND inbound_date <= @date_to
    GROUP BY {dim_groupby}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
            bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
        ],
        maximum_bytes_billed=max_bytes_billed,
    )

    df = client.query(sql, job_config=job_config, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=True)
    df["conv_rate"] = df.apply(lambda r: (r["order_cnt"] / r["ticket_cnt"]) if r["ticket_cnt"] else 0.0, axis=1)
    return df

def center_summary_from_agg(agg_df: pd.DataFrame) -> pd.DataFrame:
    cs = (
        agg_df.groupby("agent_center", dropna=False)
        .agg(
            ticket_cnt=("ticket_cnt", "sum"),
            order_cnt=("order_cnt", "sum"),
            order_amount=("order_amount", "sum"),
        )
        .reset_index()
    )
    cs["conv_rate"] = cs.apply(lambda r: (r["order_cnt"] / r["ticket_cnt"]) if r["ticket_cnt"] else 0.0, axis=1)
    return cs

# =========================================================
# 7) Raw 로드 (버튼 클릭시에만 + LIMIT)
# =========================================================
@st.cache_data(ttl=300, show_spinner=True)
def load_raw(date_from, date_to, limit_rows: int, max_bytes_billed: int) -> pd.DataFrame:
    client = get_bq_client()

    sql = f"""
    SELECT
      inbound_ts,
      inbound_date,
      request_ts,
      assigned_ts,
      ticket_id,
      inbound_channel,
      brand_name,
      matched_brand,
      agent_center,
      agent_name,
      category_lv1,
      category_lv2,
      category_lv3,
      customer_phone,
      converted_yn,
      first_order_ts,
      order_cnt,
      order_amount,
      min_leadtime_h,
      order_nos,
      sellers,
      matched_by,
      ticket_phone,
      buyer_phone,
      receiver_phone
    FROM `{SOURCE_FQN}`
    WHERE inbound_date >= @date_from
      AND inbound_date <= @date_to
    ORDER BY inbound_date, inbound_ts, ticket_id
    LIMIT @lim
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
            bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
            bigquery.ScalarQueryParameter("lim", "INT64", int(limit_rows)),
        ],
        maximum_bytes_billed=max_bytes_billed,
    )

    df = client.query(sql, job_config=job_config, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=True)
    return df

# =========================================================
# 8) 표시 포맷 (🔥 전환율 UI에서만 %로)
# =========================================================
def fmt_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _int(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return x

    def _money(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return x

    # ✅ 0.0697 -> 6.97% (UI 표시만 / 소수 2자리)
    def _rate(x):
        try:
            return f"{float(x) * 100:.2f}%"
        except Exception:
            return x

    for c in out.columns:
        if c in ["ticket_cnt", "order_cnt"]:
            out[c] = out[c].apply(_int)
        elif c == "order_amount":
            out[c] = out[c].apply(_money)
        elif c == "conv_rate":
            out[c] = out[c].apply(_rate)

    return out

# =========================================================
# 9) 실행: 집계 로드
# =========================================================
try:
    agg_df = load_agg(date_from, date_to, rows, col, max_bytes_billed)
except Exception as e:
    st.error(f"피벗 집계 로드 실패: {e}")
    st.stop()

if agg_df.empty:
    st.warning("데이터가 없습니다. (기간/필터 확인)")
    st.stop()

center_sum = center_summary_from_agg(agg_df)

total_ticket = int(center_sum["ticket_cnt"].sum())
total_orders = int(center_sum["order_cnt"].sum())
total_amount = int(center_sum["order_amount"].sum())
rate = (total_orders / total_ticket) if total_ticket else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("전체 티켓", f"{total_ticket:,}")
k2.metric("전환 주문수", f"{total_orders:,}")
k3.metric("전환 매출", f"{total_amount:,}")
k4.metric("전환율", f"{rate * 100:.2f}%")

st.divider()

# =========================================================
# 10) Tabs (피벗 / 로우데이터)
# =========================================================
tab_pivot, tab_raw = st.tabs(["📌 피벗(센터/상담사/유형)", "🧾 로우데이터 다운로드(매칭 결과)"])

# =========================================================
# (요청 2) 인입채널별 전환율: 피벗 탭 최하단 렌더
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_channel_summary(date_from, date_to, max_bytes_billed: int) -> pd.DataFrame:
    client = get_bq_client()
    sql = f"""
    SELECT
      inbound_channel,
      COUNT(1) AS ticket_cnt,
      SUM(CAST(order_cnt AS INT64)) AS order_cnt,
      SUM(CAST(order_amount AS INT64)) AS order_amount
    FROM `{SOURCE_FQN}`
    WHERE inbound_date BETWEEN @date_from AND @date_to
    GROUP BY inbound_channel
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
            bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
        ],
        maximum_bytes_billed=max_bytes_billed,
    )
    df = client.query(sql, job_config=job_cfg, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=True)
    df["inbound_channel"] = df["inbound_channel"].fillna("없음")
    df["conv_rate"] = df.apply(
        lambda r: (r["order_cnt"] / r["ticket_cnt"]) if r["ticket_cnt"] else 0.0,
        axis=1
    )
    return df

with tab_pivot:
    st.subheader("센터 요약(소계)")
    center_sum_sorted = center_sum.sort_values(sort_key, ascending=not sort_desc)

    # ✅ 화면: Rank를 index로(폭 얇게)
    center_ui_view = with_rank_index(center_sum_sorted)
    st.dataframe(
        apply_kor_columns(fmt_display(center_ui_view)),
        use_container_width=True,
        height=220,
        hide_index=False  # Rank index 표시(폭 얇음)
    )

    # ✅ CSV: Rank를 컬럼으로
    center_ui_csv = with_rank_col(center_sum_sorted)
    st.download_button(
        "센터 요약 CSV 다운로드",
        data=apply_kor_columns(center_ui_csv).to_csv(index=False).encode("utf-8-sig"),
        file_name="center_summary.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("센터별 상세 피벗")

    centers = center_sum_sorted["agent_center"].fillna("없음").tolist()

    for center in centers:
        sub = agg_df[agg_df["agent_center"] == center].copy()

        c_ticket = int(sub["ticket_cnt"].sum())
        c_orders = int(sub["order_cnt"].sum())
        c_amount = int(sub["order_amount"].sum())
        c_rate = (c_orders / c_ticket) if c_ticket else 0.0

        header = f"{center}  |  티켓 {c_ticket:,} · 전환주문 {c_orders:,} · 매출 {c_amount:,} · 전환율 {c_rate*100:.2f}%"
        with st.expander(header, expanded=(center in ["TCK", "SKMNS", "AI"])):

            if col is not None:
                values = ["ticket_cnt", "order_cnt", "order_amount", "conv_rate"]
                pv = sub.pivot_table(
                    index=rows,
                    columns=col,
                    values=values,
                    aggfunc="sum",
                    fill_value=0,
                )
                pv.columns = [f"{v} | {c}" for (v, c) in pv.columns]
                pv = pv.reset_index()
            else:
                keep_cols = ["agent_center"] + rows + ["ticket_cnt", "order_cnt", "order_amount", "conv_rate"]
                pv = sub[keep_cols].copy()

            if min_ticket > 0 and col is None and "ticket_cnt" in pv.columns:
                pv = pv[pv["ticket_cnt"] >= min_ticket]

            if col is None and sort_key in pv.columns:
                pv = pv.sort_values(sort_key, ascending=not sort_desc)

            # ✅ 화면: Rank index로(폭 얇게)
            pv_view = with_rank_index(pv)
            st.dataframe(
                apply_kor_columns(fmt_display(pv_view)),
                use_container_width=True,
                height=520,
                hide_index=False
            )

            # ✅ CSV: Rank 컬럼
            pv_csv = with_rank_col(pv)
            st.download_button(
                f"{center} 피벗 CSV 다운로드",
                data=apply_kor_columns(pv_csv).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"pivot_{center}.csv",
                mime="text/csv",
            )

    # =========================================================
    # 피벗 탭 최하단: 인입채널별 전환율 현황
    # =========================================================
    st.divider()
    st.subheader("인입채널별 전환율 현황")

    try:
        ch_df = load_channel_summary(date_from, date_to, max_bytes_billed)
    except Exception as e:
        st.error(f"인입채널별 집계 로드 실패: {e}")
        ch_df = pd.DataFrame()

    if ch_df.empty:
        st.info("인입채널 데이터가 없거나(컬럼 NULL), 기간 내 데이터가 없습니다.")
    else:
        if sort_key in ch_df.columns:
            ch_df = ch_df.sort_values(sort_key, ascending=not sort_desc)

        # ✅ 화면: Rank index로(폭 얇게)
        ch_view = with_rank_index(ch_df)
        st.dataframe(
            apply_kor_columns(fmt_display(ch_view)),
            use_container_width=True,
            height=260,
            hide_index=False
        )

        # ✅ CSV: Rank 컬럼
        ch_csv = with_rank_col(ch_df)
        st.download_button(
            "인입채널별 전환율 CSV 다운로드",
            data=apply_kor_columns(ch_csv).to_csv(index=False).encode("utf-8-sig"),
            file_name="channel_conversion_summary.csv",
            mime="text/csv",
        )

with tab_raw:
    st.subheader("로우데이터 (ticket+ order  매칭 결과 확인 / CSV 다운로드)")
    st.caption(
        "⚠️ 비용/속도 때문에 로우데이터는 **버튼을 눌렀을 때만** 불러오게끔 세팅.\n"
        f"- 기본 LIMIT: {raw_limit:,}\n"
        "- 기간이 길면 LIMIT을 올리기 전에 먼저 기간을 줄이는 걸 추천드립니다.\n"
        "- 문의 사항은 언제든지 연락 주세요 : )"
    )

    if "raw_loaded" not in st.session_state:
        st.session_state["raw_loaded"] = False
        st.session_state["raw_df"] = None

    load_btn = st.button("로우데이터 불러오기", type="primary")

    if load_btn:
        try:
            raw_df = load_raw(date_from, date_to, raw_limit, max_bytes_billed)
            st.session_state["raw_df"] = raw_df
            st.session_state["raw_loaded"] = True
        except Exception as e:
            st.error(f"로우데이터 로드 실패: {e}")
            st.stop()

    if not st.session_state["raw_loaded"]:
        st.info("⬆️ 상단 버튼을 눌러 로우데이터를 불러오세요.")
        st.stop()

    raw_df = st.session_state["raw_df"].copy()

    # ---- 필터 UI
    f1, f2, f3, f4 = st.columns(4)

    centers = sorted(raw_df["agent_center"].dropna().unique().tolist())
    agents = sorted(raw_df["agent_name"].dropna().unique().tolist())
    convs = ["O", "X"]
    matched_by_opts = sorted([x for x in raw_df["matched_by"].dropna().unique().tolist()])

    with f1:
        center_sel = st.multiselect("센터", options=centers, default=centers)
    with f2:
        agent_sel = st.multiselect("담당자", options=agents, default=agents)
    with f3:
        conv_sel = st.multiselect("전환여부", options=convs, default=convs)
    with f4:
        matched_by_sel = st.multiselect("매칭기준", options=matched_by_opts)

    q = st.text_input("검색(티켓/주문번호/전화/브랜드)", value="").strip()

    filtered = raw_df[
        raw_df["agent_center"].isin(center_sel)
        & raw_df["agent_name"].isin(agent_sel)
        & raw_df["converted_yn"].isin(conv_sel)
    ].copy()

    if matched_by_sel:
        filtered = filtered[(filtered["matched_by"].isin(matched_by_sel)) | (filtered["converted_yn"] == "X")].copy()

    if q:
        for c in ["ticket_id", "order_nos", "customer_phone", "brand_name", "matched_brand",
                  "ticket_phone", "buyer_phone", "receiver_phone", "inbound_channel"]:
            if c in filtered.columns:
                filtered.loc[:, c] = filtered[c].astype(str)

        mask = (
            filtered["ticket_id"].str.contains(q, na=False)
            | filtered["order_nos"].str.contains(q, na=False)
            | filtered["customer_phone"].str.contains(q, na=False)
            | filtered["brand_name"].str.contains(q, na=False)
            | filtered["matched_brand"].str.contains(q, na=False)
            | filtered["ticket_phone"].str.contains(q, na=False)
            | filtered["buyer_phone"].str.contains(q, na=False)
            | filtered["receiver_phone"].str.contains(q, na=False)
            | filtered["inbound_channel"].str.contains(q, na=False)
        )
        filtered = filtered[mask].copy()

    st.write(f"조회 결과: {len(filtered):,} rows (LIMIT={raw_limit:,})")

    show_cols = [
        "inbound_date", "inbound_ts",
        "ticket_id", "inbound_channel", "agent_center", "agent_name",
        "brand_name", "matched_brand",
        "category_lv1", "category_lv2", "category_lv3",
        "customer_phone",
        "converted_yn",
        "first_order_ts",
        "order_cnt", "order_amount", "min_leadtime_h",
        "order_nos", "sellers",
        "matched_by",
        "ticket_phone", "buyer_phone", "receiver_phone",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]

    raw_base = filtered[show_cols].copy()

    # ✅ 화면: Rank index로(폭 얇게)
    raw_view = with_rank_index(raw_base)
    st.dataframe(
        apply_kor_columns(raw_view),
        use_container_width=True,
        height=650,
        hide_index=False
    )

    # ✅ CSV: Rank 컬럼
    raw_csv = with_rank_col(raw_base)
    st.download_button(
        "로우데이터 CSV 다운로드(필터 반영)",
        data=apply_kor_columns(raw_csv).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"cnv_raw_{date_from}_{date_to}_limit{raw_limit}.csv",
        mime="text/csv",
    )
