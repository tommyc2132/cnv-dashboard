# cnv_dash.py
import os
import pandas as pd
import streamlit as st

from google.cloud import bigquery
from google.oauth2 import service_account

# =========================================================
# 0) 고정 설정
# =========================================================
PROJECT_ID = os.environ.get("PROJECT_ID", "strange-reducer-474905-g1").strip()

# ✅ 기본 소스: VIEW
DEFAULT_VIEW_FQN = f"{PROJECT_ID}.streamlit.01cnv"

# ✅ 비용 절감용: 파티션 테이블을 만들면 여기로 바꾸면 됨 (env로도 가능)
# 예: CNV_SOURCE=strange-reducer-474905-g1.streamlit.cnv_01cnv_tbl
SOURCE_FQN = os.environ.get("CNV_SOURCE", DEFAULT_VIEW_FQN).strip()

# ✅ 로컬용 키 경로(로컬에서만 쓰고, GitHub/배포에서는 Secrets 또는 ADC 사용)
GOOGLE_KEY_PATH = os.environ.get(
    "GOOGLE_KEY_PATH",
    r"C:\tommy\BigQuery\strange-reducer-474905-g1-946a9f4f9fac.json"
).strip()

# =========================================================
# 1) Streamlit 기본
# =========================================================
st.set_page_config(page_title="상담→주문(0~72h) 피벗 대시보드", layout="wide")
st.title("📊 상담 → 주문(0~72h) 피벗 대시보드 조영우")

# =========================================================
# 2) BigQuery Client (로컬/깃/배포 범용)
# 우선순위:
#  1) 로컬 키파일(GOOGLE_KEY_PATH 존재)  ← 로컬에서 가장 안정
#  2) Streamlit Secrets(st.secrets["gcp_service_account"])  ← Streamlit Cloud 정석
#  3) ADC (Application Default Credentials)               ← Cloud Run/로컬 gcloud auth 등
# =========================================================
@st.cache_resource(show_spinner=False)
def get_bq_client():
    # 1) 로컬 키파일 우선
    if GOOGLE_KEY_PATH and os.path.exists(GOOGLE_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=PROJECT_ID, credentials=creds)

    # 2) Secrets (secrets.toml 없으면 st.secrets 접근 자체가 예외일 수 있음 → 정상 fallback)
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return bigquery.Client(project=PROJECT_ID, credentials=creds)
    except FileNotFoundError:
        # 로컬에서 secrets.toml 없을 때 정상
        pass
    except Exception as e:
        # secrets가 "있는데" 포맷이 깨진 경우 등
        raise RuntimeError(f"Streamlit Secrets 로드/파싱 실패: {e}")

    # 3) ADC 시도 (Cloud Run / gcloud auth application-default login 등)
    try:
        return bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        raise FileNotFoundError(
            "BigQuery 인증 정보를 찾을 수 없습니다.\n\n"
            "[로컬]\n"
            "- GOOGLE_KEY_PATH 경로의 서비스계정 JSON 파일이 존재하는지 확인\n"
            "- 또는 `gcloud auth application-default login` 으로 ADC 설정\n\n"
            "[배포(Streamlit Cloud)]\n"
            "- Settings → Secrets 에 gcp_service_account 등록\n\n"
            f"원인: {e}"
        )

# =========================================================
# 3) 비용 캡(실수 방지)
# =========================================================
def bytes_from_gb(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)

# =========================================================
# 4) Pivot 집계 쿼리 (비용 최소화 핵심)
# =========================================================
@st.cache_data(ttl=300, show_spinner=True)
def load_agg(date_from, date_to, rows, col, max_bytes_billed: int) -> pd.DataFrame:
    client = get_bq_client()

    if not rows:
        raise ValueError("ROWS는 최소 1개 필요")

    # col이 rows에 들어가면 중복이므로 제거
    if col is not None and col in rows:
        col = None

    dim_fields = ["agent_center"] + rows + ([col] if col else [])
    dim_select = ",\n      ".join([f"{f}" for f in dim_fields])
    dim_groupby = ", ".join(dim_fields)

    sql = f"""
    SELECT
      {dim_select},
      COUNT(1) AS ticket_cnt,
      SUM(CAST(conv_order_cnt AS INT64)) AS conv_order_cnt,
      SUM(CAST(conv_pay_amount AS INT64)) AS conv_pay_amount
    FROM `{SOURCE_FQN}`
    WHERE inbound_date_kst >= @date_from
      AND inbound_date_kst <= @date_to
    GROUP BY {dim_groupby}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
            bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
        ],
        maximum_bytes_billed=max_bytes_billed,
    )

    df = client.query(sql, job_config=job_config).to_dataframe(create_bqstorage_client=True)

    # 전환율(엑셀 동일): SUM(conv_order_cnt) / COUNT(ticket_id)
    df["conv_rate"] = df.apply(
        lambda r: (r["conv_order_cnt"] / r["ticket_cnt"]) if r["ticket_cnt"] else 0.0, axis=1
    )
    return df

# =========================================================
# 5) KPI용(센터 요약)도 집계 결과에서 계산
# =========================================================
def center_summary_from_agg(agg_df: pd.DataFrame) -> pd.DataFrame:
    cs = (
        agg_df.groupby("agent_center", dropna=False)
        .agg(
            ticket_cnt=("ticket_cnt", "sum"),
            conv_order_cnt=("conv_order_cnt", "sum"),
            conv_pay_amount=("conv_pay_amount", "sum"),
        )
        .reset_index()
    )
    cs["conv_rate"] = cs.apply(
        lambda r: (r["conv_order_cnt"] / r["ticket_cnt"]) if r["ticket_cnt"] else 0.0, axis=1
    )
    return cs

# =========================================================
# 6) Raw 데이터 로드 (버튼 클릭 시에만 + 기본 LIMIT)
# =========================================================
@st.cache_data(ttl=300, show_spinner=True)
def load_raw(date_from, date_to, limit_rows: int, max_bytes_billed: int) -> pd.DataFrame:
    client = get_bq_client()

    sql = f"""
    SELECT
      inbound_ts_kst,
      inbound_date_kst,
      request_ts_kst,
      assigned_ts_kst,

      ticket_id,
      brand_name,
      matched_order_brand,
      agent_center,
      agent_name,
      category_lv1,
      category_lv2,
      category_lv3,
      customer_phone,

      converted_yn,
      first_order_ts_kst,
      conv_order_cnt,
      conv_pay_amount,
      min_leadtime_hours,
      conv_order_nos,
      conv_sellers,
      matched_by,

      `매칭_티켓번호`,
      `매칭_주문자번호`,
      `매칭_수령자번호`
    FROM `{SOURCE_FQN}`
    WHERE inbound_date_kst >= @date_from
      AND inbound_date_kst <= @date_to
    ORDER BY inbound_date_kst, inbound_ts_kst, ticket_id
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

    df = client.query(sql, job_config=job_config).to_dataframe(create_bqstorage_client=True)

    for c in ["inbound_ts_kst", "request_ts_kst", "assigned_ts_kst", "first_order_ts_kst"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "inbound_date_kst" in df.columns:
        df["inbound_date_kst"] = pd.to_datetime(df["inbound_date_kst"], errors="coerce").dt.date

    for c in ["conv_order_cnt", "conv_pay_amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    if "min_leadtime_hours" in df.columns:
        df["min_leadtime_hours"] = pd.to_numeric(df["min_leadtime_hours"], errors="coerce")

    df["agent_center"] = df["agent_center"].fillna("없음")
    df["agent_name"] = df["agent_name"].fillna("없음")

    return df

# =========================================================
# 7) 표시 포맷
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

    def _rate(x):
        try:
            return f"{float(x) * 100:.1f}%"
        except Exception:
            return x

    for c in out.columns:
        if "ticket_cnt" in c:
            out[c] = out[c].apply(_int)
        elif "conv_order_cnt" in c:
            out[c] = out[c].apply(_int)
        elif "conv_pay_amount" in c:
            out[c] = out[c].apply(_money)
        elif "conv_rate" in c:
            out[c] = out[c].apply(_rate)
    return out

# =========================================================
# 8) Sidebar
# =========================================================
st.sidebar.header("필터")
default_from = pd.to_datetime("2026-01-01").date()
default_to = pd.to_datetime("2026-01-31").date()
date_from = st.sidebar.date_input("시작일", value=default_from)
date_to = st.sidebar.date_input("종료일", value=default_to)

st.sidebar.caption("전환율 = SUM(conv_order_cnt) / COUNT(ticket_id) (엑셀 동일)")

st.sidebar.divider()
st.sidebar.header("피벗 설정")

available_dims = [
    "agent_name",
    "brand_name",
    "category_lv1", "category_lv2", "category_lv3",
    "matched_order_brand",
    "converted_yn",
]

rows = st.sidebar.multiselect("ROWS (드릴다운)", options=available_dims, default=["agent_name"])

col_candidates = [c for c in available_dims if c not in set(rows)]
col = st.sidebar.selectbox("COLUMNS (선택)", options=["(없음)"] + col_candidates, index=0)
col = None if col == "(없음)" else col

min_ticket = st.sidebar.number_input("최소 티켓수(필터)", min_value=0, value=0, step=10)
sort_key = st.sidebar.selectbox(
    "정렬", options=["ticket_cnt", "conv_order_cnt", "conv_pay_amount", "conv_rate"], index=3
)
sort_desc = st.sidebar.checkbox("내림차순", value=True)

st.sidebar.divider()
st.sidebar.header("BigQuery 비용 안전장치")

max_gb = st.sidebar.slider("쿼리 최대 스캔 허용(GB)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
max_bytes_billed = bytes_from_gb(max_gb)

raw_limit = st.sidebar.selectbox(
    "로우데이터 기본 LIMIT", options=[1000, 5000, 20000, 50000, 100000], index=3
)

# =========================================================
# 9) 집계 로드
# =========================================================
try:
    agg_df = load_agg(date_from, date_to, rows, col, max_bytes_billed)
except Exception as e:
    st.error(f"피벗 집계 로드 실패: {e}")
    st.stop()

if agg_df.empty:
    st.warning("데이터가 없습니다. (기간/필터 확인)")
    st.stop()

# =========================================================
# 10) KPI
# =========================================================
center_sum = center_summary_from_agg(agg_df)

total_ticket = int(center_sum["ticket_cnt"].sum())
total_conv_orders = int(center_sum["conv_order_cnt"].sum())
total_pay = int(center_sum["conv_pay_amount"].sum())
rate = (total_conv_orders / total_ticket) if total_ticket else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("전체 티켓", f"{total_ticket:,}")
k2.metric("전환 주문건수 합", f"{total_conv_orders:,}")
k3.metric("전환 매출 합", f"{total_pay:,}")
k4.metric("전환율(주문건수/티켓)", f"{rate * 100:.1f}%")

st.divider()

# =========================================================
# 11) Tabs
# =========================================================
tab_pivot, tab_raw = st.tabs(["📌 피벗", "🧾 로우데이터(Left Join 결과)"])

with tab_pivot:
    st.subheader("센터 요약(소계)")
    center_sum_sorted = center_sum.sort_values(sort_key, ascending=not sort_desc)
    st.dataframe(fmt_display(center_sum_sorted), use_container_width=True, height=220)

    st.download_button(
        "센터 요약 CSV 다운로드",
        data=center_sum_sorted.to_csv(index=False).encode("utf-8-sig"),
        file_name="center_summary.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("센터별 상세 피벗")
    centers = center_sum_sorted["agent_center"].tolist()

    for center in centers:
        sub = agg_df[agg_df["agent_center"] == center].copy()

        c_ticket = int(sub["ticket_cnt"].sum())
        c_conv = int(sub["conv_order_cnt"].sum())
        c_pay = int(sub["conv_pay_amount"].sum())
        c_rate = (c_conv / c_ticket) if c_ticket else 0.0

        header = f"{center}  |  티켓 {c_ticket:,} · 전환주문 {c_conv:,} · 매출 {c_pay:,} · 전환율 {c_rate*100:.1f}%"
        with st.expander(header, expanded=(center in ["TCK", "SKMNS", "AI"])):

            if col is not None:
                values = ["ticket_cnt", "conv_order_cnt", "conv_pay_amount", "conv_rate"]
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
                keep_cols = ["agent_center"] + rows + ["ticket_cnt", "conv_order_cnt", "conv_pay_amount", "conv_rate"]
                pv = sub[keep_cols].copy()

            if min_ticket > 0 and col is None and "ticket_cnt" in pv.columns:
                pv = pv[pv["ticket_cnt"] >= min_ticket]

            if col is None and sort_key in pv.columns:
                pv = pv.sort_values(sort_key, ascending=not sort_desc)

            st.dataframe(fmt_display(pv), use_container_width=True, height=520)

            st.download_button(
                f"{center} 피벗 CSV 다운로드",
                data=pv.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"pivot_{center}.csv",
                mime="text/csv",
            )

with tab_raw:
    st.subheader("로우데이터 (left join 결과 확인 / CSV 다운로드)")
    st.caption(
        "⚠️ 로우데이터는 비용/속도를 위해 버튼 클릭 시에만 불러오며, 기본 LIMIT가 걸어둠.\n"
        "원하면 사이드바에서 리밋 올려서 조회가능함 \n 한달치 정도는 디폴트 세팅값에서 조회해도 충분합니다."
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
        st.info("버튼을 눌러 로우데이터를 불러오세요.")
        st.stop()

    raw_df = st.session_state["raw_df"].copy()

    f1, f2, f3, f4 = st.columns(4)

    centers = sorted(raw_df["agent_center"].dropna().unique().tolist())
    agents = sorted(raw_df["agent_name"].dropna().unique().tolist())
    convs = ["O", "X"]
    matched_by_opts = sorted([x for x in raw_df["matched_by"].dropna().unique().tolist()])

    with f1:
        center_sel = st.multiselect("센터", options=centers, default=centers)
    with f2:
        agent_sel = st.multiselect("상담사", options=agents, default=agents)
    with f3:
        conv_sel = st.multiselect("전환여부(converted_yn)", options=convs, default=convs)
    with f4:
        matched_by_sel = st.multiselect("매칭기준(matched_by)", options=matched_by_opts)

    q = st.text_input("검색(ticket_id / 주문번호 / 전화번호 / 브랜드)", value="").strip()

    filtered = raw_df[
        raw_df["agent_center"].isin(center_sel)
        & raw_df["agent_name"].isin(agent_sel)
        & raw_df["converted_yn"].isin(conv_sel)
    ]

    if matched_by_sel:
        filtered = filtered[(filtered["matched_by"].isin(matched_by_sel)) | (filtered["converted_yn"] == "X")]

    if q:
        for c in [
            "ticket_id", "conv_order_nos", "customer_phone", "brand_name", "matched_order_brand",
            "매칭_티켓번호", "매칭_주문자번호", "매칭_수령자번호",
        ]:
            if c in filtered.columns:
                filtered[c] = filtered[c].astype(str)

        mask = (
            filtered["ticket_id"].str.contains(q, na=False)
            | filtered["conv_order_nos"].str.contains(q, na=False)
            | filtered["customer_phone"].str.contains(q, na=False)
            | filtered["brand_name"].str.contains(q, na=False)
            | filtered["matched_order_brand"].str.contains(q, na=False)
            | filtered["매칭_티켓번호"].str.contains(q, na=False)
            | filtered["매칭_주문자번호"].str.contains(q, na=False)
            | filtered["매칭_수령자번호"].str.contains(q, na=False)
        )
        filtered = filtered[mask]

    st.write(f"조회 결과: {len(filtered):,} rows (LIMIT={raw_limit:,})")

    show_cols = [
        "inbound_date_kst", "inbound_ts_kst",
        "ticket_id", "agent_center", "agent_name",
        "brand_name", "matched_order_brand",
        "category_lv1", "category_lv2", "category_lv3",
        "customer_phone",
        "converted_yn",
        "first_order_ts_kst",
        "conv_order_cnt", "conv_pay_amount", "min_leadtime_hours",
        "conv_order_nos", "conv_sellers",
        "matched_by",
        "매칭_티켓번호", "매칭_주문자번호", "매칭_수령자번호",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]

    st.dataframe(filtered[show_cols], use_container_width=True, height=650)

    st.download_button(
        "로우데이터 CSV 다운로드(필터 반영)",
        data=filtered[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"01cnv_raw_{date_from}_{date_to}_limit{raw_limit}.csv",
        mime="text/csv",
    )
