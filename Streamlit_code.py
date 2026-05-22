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
    merged = merged.sort_index()
    merged.index = merged.index.round("min")

    # Zorg dat meetkolommen numeriek zijn
    numeric_columns = ["eda", "prv", "temp", "hr", "resp", "movement"]

    for col in numeric_columns:
        if col in merged.columns:
            if isinstance(merged[col], pd.DataFrame):
                merged[col] = merged[col].apply(pd.to_numeric, errors="coerce")
            else:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Dubbele numerieke kolommen samenvoegen met gemiddelde
    numeric_df = merged.select_dtypes(include="number")
    numeric_df = numeric_df.T.groupby(level=0).mean().T

    # Niet-numerieke kolommen, zoals activity_intensity, apart bewaren
    non_numeric_df = merged.select_dtypes(exclude="number")

    if not non_numeric_df.empty:
        non_numeric_df = non_numeric_df.T.groupby(level=0).first().T
        merged = pd.concat([numeric_df, non_numeric_df], axis=1)
    else:
        merged = numeric_df

    return merged



def aggregate_data(df, mode):
    numeric_df = df.select_dtypes(include="number")

    if mode == "Ruwe data / per minuut":
        return numeric_df.resample("1min").mean()

    if mode == "Gemiddelde per uur":
        return numeric_df.resample("1h").mean()

    if mode == "Gemiddelde per dag":
        return numeric_df.resample("1d").mean()

    return numeric_df



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
def prv_scatter_plot(df):
    fig = go.Figure()

    data = df["prv"].dropna()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="markers",
            name="Meetpunten",
            marker=dict(color="#2ca02c", size=6, opacity=0.45)
        )
    )

    smooth = data.rolling(window=5, min_periods=1).mean()

    fig.add_trace(
        go.Scatter(
            x=smooth.index,
            y=smooth.values,
            mode="lines",
            name="Gemiddelde lijn",
            line=dict(color="#145a32", width=3)
        )
    )

    fig.update_layout(
        title="PRV / hartslagvariabiliteit",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        template="plotly_white",
        xaxis_title="Tijd",
        yaxis_title="PRV RMSSD (ms)",
        hovermode="x unified"
    )

    return fig


def activity_intensity_barplot(df):
    if "activity_intensity" not in df.columns:
        return None

    activity = df["activity_intensity"].dropna()

    if activity.empty:
        return None

    percentages = activity.value_counts(normalize=True).sort_index() * 100

    colors = [
        "#4e79a7",
        "#59a14f",
        "#f28e2b",
        "#e15759",
        "#b07aa1",
        "#76b7b2"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=percentages.index.astype(str),
            y=percentages.values,
            marker_color=colors[:len(percentages)],
            text=[f"{value:.1f}%" for value in percentages.values],
            textposition="outside"
        )
    )

    fig.update_layout(
        title="Activiteit per zone",
        height=420,
        margin=dict(l=30, r=30, t=60, b=50),
        template="plotly_white",
        xaxis_title="Activiteitszone",
        yaxis_title="Percentage van de tijd (%)",
        yaxis=dict(range=[0, max(100, percentages.max() + 10)])
    )

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
st.subheader("Grafieken")

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


if "prv" in plot_df.columns:
    st.plotly_chart(
        prv_scatter_plot(plot_df),
        use_container_width=True
    )
    explanation(
        "PRV geeft variatie tussen hartslagen weer. De losse punten zijn de ruwe meetmomenten. "
        "De lijn laat het voortschrijdend gemiddelde zien, waardoor de algemene trend beter zichtbaar wordt."
    )
else:
    st.warning("Geen PRV-data gevonden.")


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
        "Huidtemperatuur kan veranderen door lichamelijke activatie. Een daling kan soms samenhangen "
        "met spanning of enthousiasme, maar moet altijd in context worden bekeken."
    )
else:
    st.warning("Geen temperatuurdata gevonden.")


if "hr" in plot_df.columns:
    st.plotly_chart(
        line_plot(
            plot_df,
            "hr",
            "Hartslag",
            "Hartslag (BPM)",
            "#111111"
        ),
        use_container_width=True
    )
    explanation(
        "De hartslag laat zien hoeveel slagen per minuut worden gemeten. Een hogere hartslag kan passen "
        "bij beweging, spanning of emotionele activatie."
    )
else:
    st.warning("Geen hartslagdata gevonden.")


activity_fig = activity_intensity_barplot(filtered_df)

if activity_fig is not None:
    st.plotly_chart(
        activity_fig,
        use_container_width=True
    )
    explanation(
        "Deze grafiek laat zien hoeveel procent van de geselecteerde tijd de gebruiker in elke "
        "activiteitszone zat. Elke zone heeft een eigen kleur."
    )
else:
    st.warning("Geen activity intensity-data gevonden.")


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
