#streamlit code
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Data dashboard",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------
# Layout
# -------------------------------------------------
st.title("📊 Fysiologische data bekijken")
st.markdown(
    """
    Upload één of meerdere CSV-bestanden. De app herkent automatisch EDA, PRV,
    huidtemperatuur, hartslag, ademhaling en activiteit wanneer deze kolommen
    aanwezig zijn.
    """
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_single_file(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if "timestamp_iso" not in df.columns:
        st.warning(f"{uploaded_file.name} bevat geen kolom 'timestamp_iso' en is overgeslagen.")
        return None

    df["time"] = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce")
    df["time"] = df["time"].dt.tz_convert(None) + pd.Timedelta(hours=2)
    df = df.dropna(subset=["time"])
    df = df.set_index("time")
    df = df.sort_index()

    output = pd.DataFrame(index=df.index)

    if "eda_scl_usiemens" in df.columns:
        output["eda"] = df["eda_scl_usiemens"]

    if "prv_rmssd_ms" in df.columns:
        output["prv"] = df["prv_rmssd_ms"]

    if "temperature_celsius" in df.columns:
        output["temp"] = df["temperature_celsius"]

    if "pulse_rate_bpm" in df.columns:
        output["hr"] = df["pulse_rate_bpm"]

    if "respiratory_rate_brpm" in df.columns:
        output["resp"] = df["respiratory_rate_brpm"]

    if all(col in df.columns for col in ["counts_x_axis", "counts_y_axis", "counts_z_axis"]):
        x = df["counts_x_axis"]
        y = df["counts_y_axis"]
        z = df["counts_z_axis"]
        output["movement"] = np.sqrt(x**2 + y**2 + z**2)

    if "activity_intensity" in df.columns:
        output["activity_intensity"] = df["activity_intensity"]

    if output.empty:
        st.warning(f"{uploaded_file.name} bevat geen herkende meetkolommen.")
        return None

    return output


def load_uploaded_files(uploaded_files):
    dfs = []

    for uploaded_file in uploaded_files:
        df = load_single_file(uploaded_file)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, axis=1)

    # Dubbele kolommen samenvoegen door gemiddelde te nemen
    merged = merged.T.groupby(level=0).mean().T

    merged = merged.sort_index()
    merged.index = merged.index.round("min")

    return merged


def aggregate_data(df, mode):
    if mode == "Ruwe data / per minuut":
        return df.resample("1min").mean()

    if mode == "Gemiddelde per uur":
        return df.resample("1h").mean()

    if mode == "Gemiddelde per dag":
        return df.resample("1d").mean()

    return df


def filter_by_time(df, start_date, start_time, end_date, end_time):
    start = pd.to_datetime(f"{start_date} {start_time}")
    end = pd.to_datetime(f"{end_date} {end_time}")
    return df.loc[start:end]


def line_plot(df, column, title, y_label, color):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column],
            mode="lines",
            line=dict(color=color, width=2),
            name=title
        )
    )

    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=30, r=30, t=60, b=30),
        template="plotly_white",
        xaxis_title="Tijd",
        yaxis_title=y_label,
        hovermode="x unified"
    )

    return fig


def hr_activity_plot(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["hr"],
            mode="lines",
            name="Hartslag",
            line=dict(color="#111111", width=2)
        ),
        secondary_y=False
    )

    if "movement" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["movement"],
                mode="lines",
                name="Activiteit / beweging",
                line=dict(color="#7a7a7a", width=1),
                fill="tozeroy",
                opacity=0.35
            ),
            secondary_y=True
        )

    fig.update_layout(
        title="Hartslag tegenover activiteit",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        template="plotly_white",
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Tijd")
    fig.update_yaxes(title_text="Hartslag (BPM)", secondary_y=False)
    fig.update_yaxes(title_text="Activiteit / beweging", secondary_y=True)

    return fig


def explanation(text):
    st.markdown(
        f"""
        <div style="
            background-color:#f7f9fc;
            border-left:4px solid #5b8def;
            padding:0.85rem 1rem;
            border-radius:0.4rem;
            margin-top:-0.5rem;
            margin-bottom:1.5rem;
            color:#263238;
            font-size:0.95rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Bestanden")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV-bestanden",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.header("Filters")

aggregation_mode = st.sidebar.radio(
    "Weergave",
    [
        "Ruwe data / per minuut",
        "Gemiddelde per uur",
        "Gemiddelde per dag"
    ]
)


# -------------------------------------------------
# Data laden
# -------------------------------------------------
if not uploaded_files:
    st.info("Upload CSV-bestanden om te starten.")
    st.stop()

df = load_uploaded_files(uploaded_files)

if df.empty:
    st.error("Er kon geen bruikbare data worden ingeladen.")
    st.stop()


min_date = df.index.min().date()
max_date = df.index.max().date()
min_time = df.index.min().time()
max_time = df.index.max().time()

start_date = st.sidebar.date_input("Startdatum", value=min_date, min_value=min_date, max_value=max_date)
start_time = st.sidebar.time_input("Starttijd", value=min_time)

end_date = st.sidebar.date_input("Einddatum", value=max_date, min_value=min_date, max_value=max_date)
end_time = st.sidebar.time_input("Eindtijd", value=max_time)

filtered_df = filter_by_time(df, start_date, start_time, end_date, end_time)
plot_df = aggregate_data(filtered_df, aggregation_mode)

if plot_df.empty:
    st.warning("Geen data binnen het gekozen tijdsfilter.")
    st.stop()


# -------------------------------------------------
# Overzicht
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Start", plot_df.index.min().strftime("%d-%m-%Y %H:%M"))
col2.metric("Einde", plot_df.index.max().strftime("%d-%m-%Y %H:%M"))
col3.metric("Aantal meetpunten", len(plot_df))
col4.metric("Weergave", aggregation_mode)

st.divider()


# -------------------------------------------------
# Grafieken
# -------------------------------------------------
left, right = st.columns(2)

with left:
    if "eda" in plot_df.columns:
        st.plotly_chart(
            line_plot(
                plot_df,
                "eda",
                "EDA huidgeleiding",
                "EDA (µS)",
                "#1f77b4"
            ),
            use_container_width=True
        )
        explanation(
            "EDA meet huidgeleiding. Hogere waarden of pieken kunnen wijzen op verhoogde activatie, "
            "bijvoorbeeld door spanning, stress of enthousiasme."
        )
    else:
        st.warning("Geen EDA-data gevonden.")

with right:
    if "prv" in plot_df.columns:
        st.plotly_chart(
            line_plot(
                plot_df,
                "prv",
                "PRV / hartslagvariabiliteit",
                "PRV RMSSD (ms)",
                "#2ca02c"
            ),
            use_container_width=True
        )
        explanation(
            "PRV geeft variatie tussen hartslagen weer. Een hogere PRV wordt vaak geassocieerd "
            "met ontspanning en herstel, mits de signaalkwaliteit voldoende is."
        )
    else:
        st.warning("Geen PRV-data gevonden.")


left, right = st.columns(2)

with left:
    if "temp" in plot_df.columns:
        st.plotly_chart(
            line_plot(
                plot_df,
                "temp",
                "Huidtemperatuur",
                "Temperatuur (°C)",
                "#d62728"
            ),
            use_container_width=True
        )
        explanation(
            "Huidtemperatuur kan veranderen door lichamelijke activatie. Een daling kan soms "
            "samenhangen met spanning of enthousiasme, maar moet altijd in context worden bekeken."
        )
    else:
        st.warning("Geen temperatuurdata gevonden.")

with right:
    if "hr" in plot_df.columns:
        st.plotly_chart(
            hr_activity_plot(plot_df),
            use_container_width=True
        )
        explanation(
            "Deze grafiek vergelijkt hartslag met beweging. Een hogere hartslag tijdens veel beweging "
            "kan passen bij activiteit. Een hogere hartslag zonder beweging kan mogelijk wijzen op spanning "
            "of emotionele activatie."
        )
    else:
        st.warning("Geen hartslagdata gevonden.")


if "resp" in plot_df.columns:
    st.plotly_chart(
        line_plot(
            plot_df,
            "resp",
            "Ademhalingsfrequentie",
            "Ademhaling per minuut",
            "#9467bd"
        ),
        use_container_width=True
    )
    explanation(
        "De ademhalingsfrequentie laat zien hoe vaak iemand per minuut ademt. Een lagere frequentie "
        "past vaak bij rust, terwijl een hogere frequentie kan passen bij activiteit of spanning."
    )


# -------------------------------------------------
# Data bekijken
# -------------------------------------------------
st.divider()

with st.expander("Ingeladen data bekijken"):
    st.dataframe(plot_df, use_container_width=True)

with st.expander("Kolommen in deze dataset"):
    st.write(list(plot_df.columns))
