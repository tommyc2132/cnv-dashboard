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

# =========================================================
# 1) Streamlit 기본
# =========================================================
st.set_page_config(page_title="상담 → 주문(0~72h) 대시보드", layout="wide")
st.title("📊 상담 → 주문(0~72h) 대시보드")
st.caption(f"Source: `{SOURCE_FQN}` · 날짜 기준: 상담일자(inbound_date)")

# =========================================================
# 2) BigQuery Client
# =========================================================
@st.cache_resource(show_spinner=False)
def get_bq_client():
    if GOOGLE_KEY_PATH and os.path.exists(GOOGLE_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=PROJECT_ID, credentials=creds)

    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return bigquery.Client(project=PROJECT_ID, credentials=creds)
    except Exception:
        pass

    return bigquery.Client(project=PROJECT_ID)

# =========================================================
# 3) 한글 컬럼 매핑 (UI 전용)
# =========================================================
KOR_COL_MAP = {
    "inbound_date": "상담일자",
    "inbound_ts": "상담시점",

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

    "ticket_phone": "고객휴대폰_티켓",
    "buyer_phone": "주문자휴대폰",
    "receiver_phone": "수취인휴대폰",
}

def apply_kor_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in KOR_COL_MAP.items() if k in df.columns})

# =========================================================
# 4) 비용 캡
# =========================================================
def bytes_from_gb(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)

# =========================================================
# 5) Sidebar - 기간 선택
# =========================================================
st.sidebar.header("기간 선택")

today = date.today()
this_month_start = today.replace(day=1)
this_month_end = today.replace(
    day=calendar.monthrange(today.year, today.month)[1]
)

# 최근 12개월
month_options = []
for i in range(12):
    y = (this_month_start.year * 12 + this_month_start.month - 1 - i) // 12
    m = (this_month_start.month - 1 - i) % 12 + 1
    month_options.append(f"{y}-{m:02d}")

selected_month = st.sidebar.selectbox(
    "월 선택", options=month_options, index=0
)

sel_year, sel_month = map(int, selected_month.split("-"))
month_start = date(sel_year, sel_month, 1)
month_end = date(sel_year, sel_month, calendar.monthrange(sel_year, sel_month)[1])

date_from = st.sidebar.date_input("시작일", value=month_start)
date_to = st.sidebar.date_input("종료일", value=month_end)

if date_from > date_to:
    st.sidebar.error("시작일이 종료일보다 클 수 없습니다.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("BigQuery 비용 안전장치")
max_gb = st.sidebar.slider("최대 스캔 허용(GB)", 0.5, 50.0, 5.0, 0.5)
max_bytes_billed = bytes_from_gb(max_gb)

# =========================================================
# 6) 집계 쿼리
# =========================================================
@st.cache_data(ttl=300)
def load_agg(date_from, date_to, max_bytes_billed):
    client = get_bq_client()

    sql = f"""
    SELECT
      agent_center,
      COUNT(1) AS ticket_cnt,
      SUM(order_cnt) AS order_cnt,
      SUM(order_amount) AS order_amount
    FROM `{SOURCE_FQN}`
    WHERE inbound_date BETWEEN @date_from AND @date_to
    GROUP BY agent_center
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
            bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
        ],
        maximum_bytes_billed=max_bytes_billed,
    )

    df = client.query(sql, job_config=job_config).to_dataframe()
    df["conv_rate"] = df.apply(
        lambda r: (r["order_cnt"] / r["ticket_cnt"]) if r["ticket_cnt"] else 0.0,
        axis=1,
    )
    return df

# =========================================================
# 7) KPI
# =========================================================
agg_df = load_agg(date_from, date_to, max_bytes_billed)

total_ticket = int(agg_df["ticket_cnt"].sum())
total_orders = int(agg_df["order_cnt"].sum())
total_amount = int(agg_df["order_amount"].sum())
rate = (total_orders / total_ticket) if total_ticket else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("전체 티켓", f"{total_ticket:,}")
k2.metric("전환 주문수", f"{total_orders:,}")
k3.metric("전환 매출", f"{total_amount:,}")
k4.metric("전환율", f"{rate*100:.1f}%")

st.divider()

# =========================================================
# 8) Raw 데이터
# =========================================================
@st.cache_data(ttl=300)
def load_raw(date_from, date_to, max_bytes_billed):
    client = get_bq_client()

    sql = f"""
    SELECT *
    FROM `{SOURCE_FQN}`
    WHERE inbound_date BETWEEN @date_from AND @date_to
    ORDER BY inbound_date, inbound_ts, ticket_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
            bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
        ],
        maximum_bytes_billed=max_bytes_billed,
    )

    return client.query(sql, job_config=job_config).to_dataframe()

raw_df = load_raw(date_from, date_to, max_bytes_billed)

st.subheader("로우데이터 (상담 기준)")
st.dataframe(
    apply_kor_columns(raw_df),
    use_container_width=True,
    height=650,
)

st.download_button(
    "CSV 다운로드",
    data=apply_kor_columns(raw_df).to_csv(index=False).encode("utf-8-sig"),
    file_name=f"cnv_raw_{date_from}_{date_to}.csv",
    mime="text/csv",
)
