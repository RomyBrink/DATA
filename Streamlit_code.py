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
        
    if "met" in df.columns:
        output["met"] = pd.to_numeric(df["met"], errors="coerce")

    if "sleep_detection_stage" in df.columns:
        output["sleep_detection_stage"] = df["sleep_detection_stage"]


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
    numeric_columns = ["eda", "prv", "temp", "hr", "resp", "movement", "met"]

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

def met_plot(df, aggregation_mode):
    if "met" not in df.columns:
        return None

    data = df["met"].dropna()

    if data.empty:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="lines",
            name="MET",
            line=dict(color="#ff7f0e", width=2)
        )
    )

    gemiddelde = data.mean()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[gemiddelde] * len(data),
            mode="lines",
            name=f"Gemiddelde MET: {gemiddelde:.2f}",
            line=dict(color="#111111", width=2, dash="dash")
        )
    )

    fig.add_hrect(y0=0, y1=1.5, fillcolor="#4e79a7", opacity=0.10, line_width=0)
    fig.add_hrect(y0=1.5, y1=3.0, fillcolor="#59a14f", opacity=0.10, line_width=0)
    fig.add_hrect(y0=3.0, y1=6.0, fillcolor="#f28e2b", opacity=0.10, line_width=0)
    fig.add_hrect(y0=6.0, y1=max(6.1, data.max()), fillcolor="#e15759", opacity=0.10, line_width=0)

    fig.update_layout(
        title="MET: energieverbruik tijdens activiteit",
        height=440,
        margin=dict(l=30, r=30, t=60, b=40),
        template="plotly_white",
        xaxis_title="Tijd",
        yaxis_title="MET",
        hovermode="x unified",
        legend_title="Waarde"
    )

    return fig


def clean_sleep_label(value):
    value = str(value).strip()

    mapping = {
        "0": "Wakker",
        "101": "Slaap",
        "102": "Wakker tijdens slaapperiode",
        "300": "Langere ontwaking",
        "0.0": "Wakker",
        "101.0": "Slaap",
        "102.0": "Wakker tijdens slaapperiode",
        "300.0": "Langere ontwaking"
    }

    return mapping.get(value, value)


def sleep_detection_barplot(df, aggregation_mode):
    if "sleep_detection_stage" not in df.columns:
        return None

    sleep = df["sleep_detection_stage"]

    if isinstance(sleep, pd.DataFrame):
        sleep = sleep.bfill(axis=1).iloc[:, 0]

    sleep = sleep.dropna()

    if sleep.empty:
        return None

    sleep = sleep.apply(clean_sleep_label)

    if aggregation_mode == "Ruwe data / per minuut":
        freq = "1min"
        title = "Slaapdetectie per minuut"
    elif aggregation_mode == "Gemiddelde per uur":
        freq = "1h"
        title = "Slaapdetectie per uur"
    elif aggregation_mode == "Gemiddelde per dag":
        freq = "1d"
        title = "Slaapdetectie per dag"
    else:
        freq = "1min"
        title = "Slaapdetectie"

    sleep_df = pd.DataFrame(index=sleep.index)
    sleep_df["fase"] = sleep.values
    sleep_df["periode"] = sleep_df.index.floor(freq)

    counts = pd.crosstab(
        sleep_df["periode"],
        sleep_df["fase"],
        normalize="index"
    ) * 100

    if counts.empty:
        return None

    color_map = {
        "Wakker": "#4e79a7",
        "Slaap": "#59a14f",
        "Wakker tijdens slaapperiode": "#f28e2b",
        "Langere ontwaking": "#e15759"
    }

    gewenste_volgorde = [
        "Slaap",
        "Wakker",
        "Wakker tijdens slaapperiode",
        "Langere ontwaking"
    ]

    kolommen = [col for col in gewenste_volgorde if col in counts.columns]
    kolommen += [col for col in counts.columns if col not in kolommen]

    fig = go.Figure()

    for fase in kolommen:
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts[fase],
                name=fase,
                marker_color=color_map.get(fase, "#999999"),
                hovertemplate="%{x}<br>" + fase + ": %{y:.1f}%<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=30, r=30, t=60, b=50),
        template="plotly_white",
        xaxis_title="Tijd",
        yaxis_title="Percentage van de tijd (%)",
        barmode="stack",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend_title="Slaapfase"
    )

    return fig


def scatter_with_average_plot(df, column, title, y_label, color):
    fig = go.Figure()

    data = df[column].dropna()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="markers",
            name="Meetpunten",
            marker=dict(color=color, size=6, opacity=0.45)
        )
    )

    smooth = data.rolling(window=5, min_periods=1).mean()

    fig.add_trace(
        go.Scatter(
            x=smooth.index,
            y=smooth.values,
            mode="lines",
            name="Gemiddelde lijn",
            line=dict(color=color, width=3)
        )
    )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        template="plotly_white",
        xaxis_title="Tijd",
        yaxis_title=y_label,
        hovermode="x unified"
    )

    return fig


def clean_activity_label(value):
    mapping = {
        "sedentary": "Stilstand",
        "Sedentary": "Stilstand",
        "SEDENTARY": "Stilstand",
        "VPA": "Zware inspanning",
        "MPA": "Matige inspanning",
        "LPA": "Lichte inspanning"
    }

    return mapping.get(str(value), str(value))


def activity_intensity_barplot(df, aggregation_mode):
    if "activity_intensity" not in df.columns:
        return None

    activity = df["activity_intensity"].dropna()

    if activity.empty:
        return None

    activity = activity.apply(clean_activity_label)

    if aggregation_mode == "Ruwe data / per minuut":
        freq = "1min"
        title = "Activiteit per zone per minuut"
    elif aggregation_mode == "Gemiddelde per uur":
        freq = "1h"
        title = "Activiteit per zone per uur"
    elif aggregation_mode == "Gemiddelde per dag":
        freq = "1d"
        title = "Activiteit per zone per dag"
    else:
        freq = "1min"
        title = "Activiteit per zone"

    activity_df = pd.DataFrame({"activity_intensity": activity})
    activity_df["periode"] = activity_df.index.floor(freq)

    percentages = (
        activity_df
        .groupby("periode")["activity_intensity"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percentage")
        .reset_index()
    )

    if percentages.empty:
        return None

    color_map = {
        "Stilstand": "#4e79a7",
        "Lichte inspanning": "#59a14f",
        "Matige inspanning": "#f28e2b",
        "Zware inspanning": "#e15759"
    }

    fig = go.Figure()

    for zone in percentages["activity_intensity"].unique():
        zone_df = percentages[percentages["activity_intensity"] == zone]

        fig.add_trace(
            go.Bar(
                x=zone_df["periode"],
                y=zone_df["percentage"],
                name=zone,
                marker_color=color_map.get(zone, "#999999"),
                text=[f"{value:.1f}%" for value in zone_df["percentage"]],
                textposition="inside"
            )
        )

    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=30, r=30, t=60, b=50),
        template="plotly_white",
        xaxis_title="Tijd",
        yaxis_title="Percentage van de tijd (%)",
        barmode="stack",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend_title="Activiteitszone"
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
            line=dict(color="#FF0000", width=2)
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
    st.info("Upload CSV-bestanden om te starten. Alleen de volgende data kan worden ingeladen: eda, prv, temperature_celsius, pulse_rate_bpm, respiratory_rate_brpm, activity_intensity, met en sleep_detection_stage")
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
        scatter_with_average_plot(
            plot_df,
            "prv",
            "PRV / hartslagvariabiliteit",
            "PRV RMSSD (ms)",
            "#2ca02c"
        ),
        use_container_width=True
    )
    explanation(
        "PRV geeft variatie tussen hartslagen weer. De losse punten zijn de meetmomenten. "
        "De lijn laat het gemiddelde zien. Een hogere PRV staat gelijk aan meer rust en ontspanning. Een lagere PRV kan veroorzaakt worden door activatie van het lichaam (tijdens en na inspanningen, slechte slaap, actief immuunsysteem, stress, enthousiasme)"
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
        "met acute stress. Als de huidtemperatuur gedurende langere tijd stijgt, kan dit duiden op activatie van het afweersysteem (bijvoorbeeld door virussen, koorts of ontstekingen) of op herstel na inspanning. Bij vrouwen stijgt de huidtemperatuur vaak na de eisprong."
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
            "#e60000"
        ),
        use_container_width=True
    )
    explanation(
        "De hartslag laat zien hoeveel slagen per minuut worden gemeten."
    )
else:
    st.warning("Geen hartslagdata gevonden.")


activity_fig = activity_intensity_barplot(filtered_df, aggregation_mode)

if activity_fig is not None:
    st.plotly_chart(
        activity_fig,
        use_container_width=True
    )
    explanation(
        "Deze grafiek laat per gekozen tijdseenheid zien hoeveel procent van de tijd de gebruiker in elke "
        "activiteitszone zat. Bij uurweergave wordt dit dus per uur berekend, bij dagweergave per dag."
    )
else:
    st.warning("Geen activity intensity-data gevonden.")


met_fig = met_plot(plot_df, aggregation_mode)

if met_fig is not None:
    st.plotly_chart(
        met_fig,
        use_container_width=True
    )
    explanation(
        "MET staat voor Metabolic Equivalent of Task en geeft aan hoeveel energie een activiteit kost "
        "ten opzichte van rust. Ongeveer 1 MET is rust, 1.5 tot 3 MET is lichte activiteit, 3 tot 6 MET "
        "matige activiteit en boven 6 MET zware activiteit. Deze informatie kan helpen om het dagelijkse "
        "energieverbruik beter te begrijpen. Zo kan je berekenen hoeveel calorieën je moet binnen krijgen op een dag. "
        "Doe hiervoor het geweicht x de gemiddelde MET x het aantal uur (als je aantal calorieën op 1 dag wilt berekenen dan dus x 24)"
        "Voor gewichtsbehoud eet je ongeveer evenveel calorieën als uit deze berekening komt. Voor aankomen eet je meer, en voor afvallen minder."
    )
else:
    st.warning("Geen MET-data gevonden.")


sleep_fig = sleep_detection_barplot(filtered_df, aggregation_mode)

if sleep_fig is not None:
    st.plotly_chart(
        sleep_fig,
        use_container_width=True
    )
    explanation(
        "Deze grafiek laat per gekozen tijdseenheid zien welk percentage van de tijd iemand wakker was, "
        "sliep, wakker was tijdens een slaapperiode of een langere ontwaking had. Bij uurweergave wordt "
        "dit dus per uur berekend, bij dagweergave per dag."
    )
else:
    st.warning("Geen sleep detection-data gevonden.")


if "resp" in plot_df.columns:
    st.plotly_chart(
        scatter_with_average_plot(
            plot_df,
            "resp",
            "Ademhalingsfrequentie",
            "Ademhaling per minuut",
            "#9467bd"
        ),
        use_container_width=True
    )
    explanation(
        "De ademhalingsfrequentie wordt weergegeven als losse meetpunten met een gemiddelde lijn. "
        "Zo blijft de spreiding zichtbaar, terwijl de trend makkelijker te volgen is."
    )
else:
    st.warning("Geen ademhalingsdata gevonden.")

# -------------------------------------------------
# Data bekijken
# -------------------------------------------------
st.divider()

with st.expander("Ingeladen data bekijken"):
    st.dataframe(plot_df, use_container_width=True)

with st.expander("Kolommen in deze dataset"):
    st.write(list(plot_df.columns))
