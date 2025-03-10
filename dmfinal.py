import streamlit as st
import pandas as pd
import altair as alt
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# ------------------
# 1. Load the Dataset
# ------------------
@st.cache_data
def load_data(file_path):
    """
    Load the CSV file from the given file path.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Ensure the file exists.")
        return None

# Helper function to compute technical indicators
def compute_technical_indicators(data, indicators):
    """
    Compute selected technical indicators in-place.
    Available indicators: SMA(10), SMA(50), EMA(10), EMA(50),
                          MACD, RSI, Bollinger Bands
    """
    # Make sure the data is sorted by Date
    data = data.sort_values('Date').copy()

    # Close price series for easier referencing
    close = data['Close']

    # ----- SMA(10) -----
    if "SMA(10)" in indicators:
        data['SMA10'] = close.rolling(window=10).mean()

    # ----- SMA(50) -----
    if "SMA(50)" in indicators:
        data['SMA50'] = close.rolling(window=50).mean()

    # ----- EMA(10) -----
    if "EMA(10)" in indicators:
        data['EMA10'] = close.ewm(span=10, adjust=False).mean()

    # ----- EMA(50) -----
    if "EMA(50)" in indicators:
        data['EMA50'] = close.ewm(span=50, adjust=False).mean()

    # ----- MACD (12, 26, 9) -----
    if "MACD" in indicators:
        data['EMA12'] = close.ewm(span=12, adjust=False).mean()
        data['EMA26'] = close.ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # ----- RSI (14) -----
    if "RSI" in indicators:
        # Price changes
        data['diff'] = close.diff(1)
        data['gain'] = data['diff'].apply(lambda x: x if x > 0 else 0)
        data['loss'] = data['diff'].apply(lambda x: -x if x < 0 else 0)
        data['avg_gain'] = data['gain'].rolling(14).mean()
        data['avg_loss'] = data['loss'].rolling(14).mean()
        data['rs'] = data['avg_gain'] / data['avg_loss']
        data['RSI'] = 100 - (100 / (1 + data['rs']))

    # ----- Bollinger Bands (20, 2) -----
    if "Bollinger Bands" in indicators:
        data['BB_MA'] = close.rolling(window=20).mean()
        data['BB_STD'] = close.rolling(window=20).std()
        data['BB_Upper'] = data['BB_MA'] + (2 * data['BB_STD'])
        data['BB_Lower'] = data['BB_MA'] - (2 * data['BB_STD'])

    return data

# ------------------
# 2. Main App
# ------------------
# File path for the dataset
file_path = "dataset-miniproject.csv"
df = load_data(file_path)

st.title("Admin Dashboard")

# ------------------
# 3. Sidebar Navigation
# ------------------
st.sidebar.title("Dashboard Menu")
menu_option = st.sidebar.radio(
    "Select Page:",
    ["Home", "View Dataset", "Visualization", "About"]
)

# ------------------
# 4. HOME
# ------------------
if menu_option == "Home":
    st.header("Welcome to the Admin Dashboard")
    st.write("Use the menu on the left to navigate through the dashboard features.")
    st.image("https://via.placeholder.com/800x200?text=Admin+Dashboard")  # Placeholder image

# ------------------
# 5. VIEW DATASET
# ------------------
elif menu_option == "View Dataset":
    if df is not None:
        st.header("View Dataset")
        st.write("Manage the dataset display using the options below.")

        # Checkbox to toggle dataset view
        if st.checkbox("Show Dataset Table"):
            st.dataframe(df)  # Display the dataset as a table

        # Checkbox for dataset details
        if st.checkbox("Show Dataset Details"):
            st.write(f"**Number of Rows:** {df.shape[0]}")
            st.write(f"**Number of Columns:** {df.shape[1]}")
            st.write("**Columns:**")
            st.write(", ".join(df.columns))

        # Checkbox for a download option
        if st.checkbox("Enable Dataset Download"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Dataset as CSV",
                data=csv,
                file_name='ds1.csv',
                mime='text/csv'
            )
    else:
        st.error("Dataset could not be loaded. Please check the file path.")

# ------------------
# 6. VISUALIZATION
# ------------------
elif menu_option == "Visualization":
    if df is None:
        st.error("Dataset could not be loaded. Please check the file path.")
    else:
        st.header("Visualization")
        # Sub-menu for visualization options
        vis_option = st.selectbox(
            "Choose a Visualization Option:",
            ["Industry Analysis", "Technical Indicators"]
        )

        # ================
        # 6A. INDUSTRY ANALYSIS
        # ================
        if vis_option == "Industry Analysis":
            st.subheader("Industry Analysis")

            # Convert Date column to datetime (if not already)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Drop invalid or missing dates
            df.dropna(subset=['Date'], inplace=True)

            # Ensure required columns
            required_cols_va = ["Industry_Tag", "Volume", "Date"]
            missing_cols_va = [col for col in required_cols_va if col not in df.columns]

            if missing_cols_va:
                st.error(f"The dataset must contain the following columns: {', '.join(missing_cols_va)}")
            else:
                # --- Section 1: Total Volume by Industry on a Selected Date ---
                st.write("### 1. Industry Volume by Specific Date")

                # Group data by Date and Industry, summing up Volume
                grouped_data = df.groupby(['Date', 'Industry_Tag'])['Volume'].sum().reset_index()

                # Unique dates for selection
                unique_dates = sorted(grouped_data['Date'].unique())

                # Date selection
                selected_date = st.selectbox("Select a Date:", unique_dates)

                # Filter for the selected date
                date_filtered_data = grouped_data[grouped_data['Date'] == selected_date]

                if date_filtered_data.empty:
                    st.warning(f"No data available for the selected date: {selected_date}.")
                else:
                    # Bar chart (Altair)
                    bar_chart = alt.Chart(date_filtered_data).mark_bar().encode(
                        x=alt.X("Industry_Tag", sort='-y', title="Industry Tag"),
                        y=alt.Y("Volume", title="Total Volume"),
                        tooltip=["Industry_Tag", "Volume"]
                    ).properties(
                        width=800,
                        height=400,
                        title=f"Total Volume by Industry on {selected_date.date()}"
                    )

                    st.altair_chart(bar_chart, use_container_width=True)

                # --- Section 2: Volume Trend for a Selected Industry ---
                st.write("### 2. Volume Trend for a Selected Industry")

                unique_industries = sorted(grouped_data['Industry_Tag'].unique())
                selected_industry = st.selectbox("Select an Industry:", unique_industries)

                # Filter data for the selected industry
                industry_data = grouped_data[grouped_data['Industry_Tag'] == selected_industry]

                if industry_data.empty:
                    st.warning(f"No data available for the selected industry: {selected_industry}.")
                else:
                    # Determine date range for slider
                    min_date = industry_data['Date'].min().to_pydatetime()
                    max_date = industry_data['Date'].max().to_pydatetime()

                    # Date range slider
                    date_range = st.slider(
                        "Select Date Range:",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date)
                    )

                    # Filter based on date range
                    range_filtered_data = industry_data[
                        (industry_data['Date'] >= date_range[0]) & (industry_data['Date'] <= date_range[1])
                    ]

                    # Line chart for selected industry
                    line_chart = alt.Chart(range_filtered_data).mark_line(color='green').encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Volume", title="Volume"),
                        tooltip=["Date:T", "Volume"]
                    ).properties(
                        width=800,
                        height=400,
                        title=f"Volume Trend for {selected_industry} ({date_range[0].date()} to {date_range[1].date()})"
                    )

                    st.altair_chart(line_chart, use_container_width=True)

                # --- Section 3: Volume Trends for All Industries ---
                st.write("### 3. Volume Trends for All Industries")

                all_min_date = grouped_data['Date'].min().to_pydatetime()
                all_max_date = grouped_data['Date'].max().to_pydatetime()

                date_range_all = st.slider(
                    "Select Date Range for All Industries:",
                    min_value=all_min_date,
                    max_value=all_max_date,
                    value=(all_min_date, all_max_date),
                    key="all_sectors_slider"
                )

                # Filter for all industries in the selected date range
                all_filtered_data = grouped_data[
                    (grouped_data['Date'] >= date_range_all[0]) & (grouped_data['Date'] <= date_range_all[1])
                ]

                # Line chart for all industries
                all_sectors_chart = alt.Chart(all_filtered_data).mark_line().encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Volume", title="Total Volume"),
                    color=alt.Color("Industry_Tag", title="Industry"),
                    tooltip=["Date:T", "Industry_Tag", "Volume"]
                ).properties(
                    width=900,
                    height=500,
                    title=f"Volume Trends for All Industries "
                          f"({date_range_all[0].date()} to {date_range_all[1].date()})"
                )

                st.altair_chart(all_sectors_chart, use_container_width=True)

        # ================
        # 6B. TECHNICAL INDICATORS
        # ================
        elif vis_option == "Technical Indicators":
            st.subheader("Technical Indicators")

            # Ensure required columns
            required_cols_ti = [
                "Date", "Open", "High", "Low", "Close", 
                "Volume", "Ticker", "Brand_Name", "Industry_Tag"
            ]
            missing_cols_ti = [col for col in required_cols_ti if col not in df.columns]

            if missing_cols_ti:
                st.error(f"The dataset must contain the following columns: {', '.join(missing_cols_ti)}")
            else:
                # Convert Date column to datetime
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)

                # Combine brand name and ticker for display
                df["Display"] = df["Brand_Name"] + " (" + df["Ticker"] + ")"
                unique_display_options = df["Display"].unique()

                # Select a particular stock
                selected_display = st.selectbox("Select a Stock to Visualize:", unique_display_options)
                selected_ticker = selected_display.split(" (")[1][:-1]
                ticker_data = df[df["Ticker"] == selected_ticker].copy()

                # Date range selection
                min_date = ticker_data["Date"].min().to_pydatetime()
                max_date = ticker_data["Date"].max().to_pydatetime()
                date_range = st.slider(
                    "Select Date Range for Analysis:",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date)
                )
                # Filter data
                ticker_data = ticker_data[
                    (ticker_data["Date"] >= date_range[0]) & (ticker_data["Date"] <= date_range[1])
                ].sort_values("Date")

                # -------------------------------------
                # Multi-select for Technical Indicators
                # -------------------------------------
                indicator_options = [
                    "SMA(10)",
                    "SMA(50)",
                    "EMA(10)",
                    "EMA(50)",
                    "MACD",
                    "RSI",
                    "Bollinger Bands"
                ]
                selected_indicators = st.multiselect(
                    "Select Technical Indicators:",
                    options=indicator_options,
                    default=[]
                )

                # Compute selected indicators
                if selected_indicators:
                    ticker_data = compute_technical_indicators(ticker_data, selected_indicators)
                
                # ---------------------
                # Create Subplots (Plotly)
                # ---------------------
                # We will create up to 3 rows:
                #  1) Candlestick + Overlays (SMA, EMA, Bollinger)
                #  2) MACD (if selected)
                #  3) RSI (if selected)
                # Determine how many rows are needed
                row_count = 1
                if "MACD" in selected_indicators:
                    row_count += 1
                if "RSI" in selected_indicators:
                    row_count += 1

                fig = make_subplots(
                    rows=row_count,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    subplot_titles=["Candlestick with Overlays"] 
                                   + (["MACD"] if "MACD" in selected_indicators else [])
                                   + (["RSI"] if "RSI" in selected_indicators else [])
                )

                # Row index to track which row is used
                current_row = 1

                # --- CANDLESTICK (Row 1) ---
                fig.add_trace(
                    go.Candlestick(
                        x=ticker_data["Date"],
                        open=ticker_data["Open"],
                        high=ticker_data["High"],
                        low=ticker_data["Low"],
                        close=ticker_data["Close"],
                        name="Candlestick",
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ),
                    row=current_row, col=1
                )

                # Overlays (SMA, EMA, Bollinger) on Row 1
                # --- SMA/EMA lines ---
                for ind in ["SMA10", "SMA50", "EMA10", "EMA50"]:
                    if ind in ticker_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=ticker_data["Date"],
                                y=ticker_data[ind],
                                mode="lines",
                                name=ind
                            ),
                            row=current_row, col=1
                        )

                # --- Bollinger Bands (Upper/Lower) ---
                if "Bollinger Bands" in selected_indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=ticker_data["Date"],
                            y=ticker_data["BB_Upper"],
                            mode="lines",
                            line=dict(color='gray', width=1),
                            name="BB Upper"
                        ),
                        row=current_row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=ticker_data["Date"],
                            y=ticker_data["BB_Lower"],
                            mode="lines",
                            line=dict(color='gray', width=1),
                            name="BB Lower"
                        ),
                        row=current_row, col=1
                    )

                # --- MACD (Row 2 if selected) ---
                if "MACD" in selected_indicators:
                    current_row += 1
                    fig.add_trace(
                        go.Scatter(
                            x=ticker_data["Date"],
                            y=ticker_data["MACD"],
                            line=dict(color='blue'),
                            name="MACD"
                        ),
                        row=current_row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=ticker_data["Date"],
                            y=ticker_data["MACD_Signal"],
                            line=dict(color='red'),
                            name="MACD Signal"
                        ),
                        row=current_row, col=1
                    )
                    # MACD Histogram
                    fig.add_trace(
                        go.Bar(
                            x=ticker_data["Date"],
                            y=ticker_data["MACD_Hist"],
                            marker_color='gray',
                            name="MACD Hist"
                        ),
                        row=current_row, col=1
                    )

                # --- RSI (Row 3 if selected) ---
                if "RSI" in selected_indicators:
                    current_row += 1
                    fig.add_trace(
                        go.Scatter(
                            x=ticker_data["Date"],
                            y=ticker_data["RSI"],
                            line=dict(color='orange'),
                            name="RSI"
                        ),
                        row=current_row, col=1
                    )
                    # Optional: Add lines for RSI thresholds (30/70)
                    fig.add_hrect(
                        y0=70, y1=70, line_width=1, line_dash="dash", 
                        line_color="red", fillcolor="red", opacity=0.2,
                        row=current_row, col=1
                    )
                    fig.add_hrect(
                        y0=30, y1=30, line_width=1, line_dash="dash", 
                        line_color="green", fillcolor="green", opacity=0.2,
                        row=current_row, col=1
                    )

                fig.update_layout(
                    title=f"Technical Indicators: {selected_display}",
                    xaxis_title="Date",
                    template="plotly_white",
                    width=900,
                    height=500 if row_count == 1 else (300*row_count)
                )

                fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
                fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

                st.plotly_chart(fig, use_container_width=True)

                # Optionally compare with Industry Volume (Checkbox)
                compare_with_industry = st.checkbox("Compare with Industry Volume")
                if compare_with_industry:
                    selected_industry = df[df["Ticker"] == selected_ticker]["Industry_Tag"].iloc[0]
                    industry_data = df[df["Industry_Tag"] == selected_industry].copy()
                    # Group industry data by date and sum volume
                    industry_data_grouped = industry_data.groupby("Date")["Volume"].sum().reset_index()
                    # Filter for the selected date range
                    ind_filtered = industry_data_grouped[
                        (industry_data_grouped["Date"] >= date_range[0]) &
                        (industry_data_grouped["Date"] <= date_range[1])
                    ]

                    # Compare volumes with Altair
                    st.write(f"### Volume Comparison: {selected_display} vs. Industry ({selected_industry})")

                    # Industry line chart
                    industry_line_chart = alt.Chart(ind_filtered).mark_line(color="blue").encode(
                        x=alt.X("Date:T", title="Date (Industry)"),
                        y=alt.Y("Volume", title="Industry Volume"),
                        tooltip=["Date:T", "Volume"]
                    ).properties(
                        width=900,
                        height=300,
                        title="Industry Volume Over Time"
                    )

                    # Stock volume line
                    stock_line_chart = alt.Chart(ticker_data).mark_line(color="green").encode(
                        x=alt.X("Date:T", title="Date (Stock)"),
                        y=alt.Y("Volume", title="Stock Volume"),
                        tooltip=["Date:T", "Volume"]
                    ).properties(
                        width=900,
                        height=300,
                        title="Stock Volume Over Time"
                    )

                    combined_chart = alt.layer(
                        industry_line_chart,
                        stock_line_chart
                    ).resolve_scale(y='independent').properties(
                        width=900,
                        height=400
                    )

                    st.altair_chart(combined_chart, use_container_width=True)

# ------------------
# 7. ABOUT
# ------------------
elif menu_option == "About":
    st.header("About This Dashboard")
    st.write("This dashboard is designed to help visualize and analyze stock data across various industries.")
    st.write("Use the sidebar to navigate between different sections and features.")
