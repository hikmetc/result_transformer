# Deveoper: Hikmet Can Çubukçu
# Date: 06.04.2025

import streamlit as st
st.set_page_config(page_title="Result Transformer", page_icon="🛠️")
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats  # ← new



with st.sidebar:
    with open('./template/template_data_fasting_glucose.xlsx', "rb") as template_file:
        template_byte = template_file.read()
    # download template excel file
    st.download_button(label="Click to Download Template File",
                        data=template_byte,
                        file_name="template.xlsx",
                        mime='application/octet-stream')   
    
    # Upload file widgets
    uploaded_file_source = st.file_uploader(
        '#### **Upload your source method results .xlsx (Excel) or .csv file:**',
        type=['csv', 'xlsx'],
        accept_multiple_files=False
    )

    uploaded_file_target = st.file_uploader(
        '#### **Upload your target method results .xlsx (Excel) or .csv file:**',
        type=['csv', 'xlsx'],
        accept_multiple_files=False
    )

    # Define a cached function to process file data
    @st.cache_data
    def load_data(file):        
        # Load the uploaded file (Excel or CSV)
        try:
            df = pd.read_excel(file)
        except:
            file.seek(0)  # Reset file pointer before reading again
            df = pd.read_csv(file, sep=None, engine='python')
        return df

    # Check if a source file has been uploaded
    if uploaded_file_source is not None:
        # Load the data using the cached function
        uploaded_data_source = load_data(uploaded_file_source)

        # Display the selectbox widget outside the cached function
        analyte_name_box_source = st.selectbox("**Select the Measurand Name (Source Method)**", tuple(uploaded_data_source.columns))

        # Process the selected analyte data
        analyte_data_source = uploaded_data_source[analyte_name_box_source].dropna().reset_index(drop=True)

    # Check if a target file has been uploaded
    if uploaded_file_target is not None:
        # Load the data using the cached function
        uploaded_data_target = load_data(uploaded_file_target)

        # Display the selectbox widget outside the cached function
        analyte_name_box_target = st.selectbox("**Select the Measurand Name (Target Method)**", tuple(uploaded_data_target.columns))

        # Process the selected analyte data
        analyte_data_target = uploaded_data_target[analyte_name_box_target].dropna().reset_index(drop=True)
    
    st.info('*Developed by Hikmet Can Çubukçu, MD, PhD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')




def plot_interactive_hist_density(data: pd.Series, title: str = 'Interactive Histogram and Density Plot', bins: int = None):
    """
    Plots an interactive histogram with an overlaid density plot using Plotly.

    Parameters:
    data (pd.Series): The dataset to plot.
    title (str): Title of the plot.
    bins (int, optional): Number of bins for the histogram. If None, the optimal number of bins will be calculated.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Data should be a Pandas Series.")
    
    if data.empty:
        st.warning("The provided data series is empty. Please check your input.")
        return
    
    # Determine optimal number of bins using Freedman-Diaconis rule if bins not provided
    if bins is None:
        q75, q25 = np.percentile(data.dropna(), [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * (len(data.dropna()) ** (-1/3))
        bins = max(1, int((data.max() - data.min()) / bin_width)) if bin_width > 0 else 10
    
    # Create histogram
    fig = px.histogram(
        data_frame=data.to_frame(),
        x=data.name,
        nbins=bins,
        marginal="rug",  # Add a rug plot
        opacity=0.6,
        histnorm='density',  # Normalize histogram
        title=title
    )
    
    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title=data.name,
        yaxis_title='Density',
        template='plotly_white'
    )
    
    st.plotly_chart(fig)




def plot_interactive_hist_density(data: pd.Series, title: str = 'Interactive Histogram and Density Plot', bins: int = None, x_max_limit: int = None):
    """
    Plots an interactive histogram with an overlaid density plot using Plotly.

    Parameters:
    data (pd.Series): The dataset to plot.
    title (str): Title of the plot.
    bins (int, optional): Number of bins for the histogram. If None, the optimal number of bins will be calculated.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Data should be a Pandas Series.")
    
    if data.empty:
        st.warning("The provided data series is empty. Please check your input.")
        return
    
    # Ensure data has a name for labeling
    data = data.rename(data.name if data.name else "Value")
    
    # Determine optimal number of bins using Freedman-Diaconis rule if bins not provided
    if bins is None:
        q75, q25 = np.percentile(data.dropna(), [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * (len(data.dropna()) ** (-1/3))
        bins = max(1, int((data.max() - data.min()) / bin_width)) if bin_width > 0 else 10
    
    # Create histogram
    fig = px.histogram(
        data_frame=pd.DataFrame({data.name: data}),
        x=data.name,
        nbins=bins,
        marginal="rug",  # Add a rug plot
        opacity=0.6,
        histnorm='density',  # Normalize histogram
        title=title
    )
    # Apply x-axis limit if provided
    if x_max_limit is not None:
        fig.update_xaxes(range=[data.min(), x_max_limit])

    # Update layout for better aesthetics
    fig.update_layout(
        title=dict(text=title, font=dict(color='rgb(128, 0, 32)', size=18)),  # Burgundy title
        xaxis_title=data.name,
        yaxis_title='Density',
        template='plotly_white'
    )
    
    st.plotly_chart(fig)

st.image('./images/Result Transformer-4.png')

tab1, tab2, tab3, tab4 = st.tabs(["📖 **Instructions**", "📊 **:green[Distribution and percentiles of uploaded data]**", "📊 **:orange[KS test and rescaled distributions]**", 
                                "🛠️ **:blue[Result Transformer]**"],)
with tab1:
    st.write("This is the instructions tab")
    st.markdown("""            
                #### :blue[Instructions]
                
                **If you already have the percentiles of the data from the target and source method results, 
                you can use the calculator in the tab 3 to calculate the adjusted result of the source method.**

                1. Enter the lower and upper percentiles of the data from the target method.
                
                2. Enter the lower and upper percentiles of the data from the source method.
                
                3. Enter the source method result.

                4. Click on the "Calculate Result Source" button.


                **If you do not have the percentiles of the data of target and source method results, 
                but have the data of the target and source method results, you can use this app to determine 
                percentiles of your data and then calculate the adjusted result of the source method.**

                1. Upload your data of target and source method results seperately. (e.g. template.xlsx) 
                Make sure that the first row of the Excel file you upload has measurand names and the other rows 
                have analyte values, like the following example:
                
                  | Cancer Antigen 125 | Cancer Antigen 19-9 |
                  | ----------- | ----------- |
                  | 20 | 15 |
                  | 45 | 7 |
                  | 15 | 25 |
                  | 30 | 35 |        

                2. Select the measurand name. 
                (e.g., for template.xlsx file, "Fasting Glucose (mg/dL)")
                
                3. The uploaded data will be displayed as a table and a histogram with a density plot.

                4. Enter the lower and upper percentiles of the data from the target method.
                
                5. Enter the lower and upper percentiles of the data from the source method.
                
                6. Enter the source method result.

                7. Click on the "Calculate Adjusted Source Method Result" button.

                """) 

with tab2:
    st.markdown("#### :blue[The distribution of source method results]")
    if uploaded_file_source is not None:
        # Calculate percentiles
        percentile_99_source = np.percentile(analyte_data_source, 99)
        plot_interactive_hist_density(analyte_data_source, x_max_limit=percentile_99_source)

        percentile_2_5_source = np.percentile(analyte_data_source, 2.5)
        percentile_97_5_source = np.percentile(analyte_data_source, 97.5)

        st.markdown(f"**:blue[2.5th Percentile:] {percentile_2_5_source:.3f}**")
        st.markdown(f"**:blue[97.5th Percentile:] {percentile_97_5_source:.3f}**")
    else:
        st.info("Please upload a source file to view the distribution and percentiles.")

    st.write("---")

    st.markdown("#### :blue[The distribution of target method results]")
    if uploaded_file_target is not None:
        # Calculate percentiles
        percentile_99_target = np.percentile(analyte_data_target, 99)
        plot_interactive_hist_density(analyte_data_target, x_max_limit=percentile_99_target)

        percentile_2_5_target = np.percentile(analyte_data_target, 2.5)
        percentile_97_5_target = np.percentile(analyte_data_target, 97.5)

        st.markdown(f"**:blue[2.5th Percentile:] {percentile_2_5_target:.3f}**")
        st.markdown(f"**:blue[97.5th Percentile:] {percentile_97_5_target:.3f}**")
    else:
        st.info("Please upload a target file to view the distribution and percentiles.")


with tab3:
    # ----------------------------------------------------------------------
    # NEW SECTION: KS TEST ON ROBUSTLY STANDARDIZED DATA (Using Sampled Data)
    # ----------------------------------------------------------------------

    st.markdown("#### :blue[Rescaling & Distribution Shape Comparison]")
    if (uploaded_file_source is not None) and (uploaded_file_target is not None):

        # 1) Robust standardization helper
        def robust_standardize(series: pd.Series) -> pd.Series:
            med = np.median(series)
            iqr = np.percentile(series, 75) - np.percentile(series, 25)
            if iqr == 0:
                iqr = 1e-9
            return (series - med) / iqr

        # 2) Freedman–Diaconis bin‐count helper
        def fd_bins(series: pd.Series) -> int:
            q75, q25 = np.percentile(series.dropna(), [75, 25])
            iqr = q75 - q25
            n = series.dropna().size
            bw = 2 * iqr * (n ** (-1/3))
            if bw <= 0:
                return 10
            return max(1, int(np.ceil((series.max() - series.min()) / bw)))

        # 3) Standardize both datasets
        source_std = robust_standardize(analyte_data_source).rename("Source data (resclaled)")
        target_std = robust_standardize(analyte_data_target).rename("Target data (resclaled)")

        # 4) Compute bins and 99th‐percentile caps for full standardized data
        bins_source = fd_bins(source_std)
        bins_target = fd_bins(target_std)
        bins_overlay = max(bins_source, bins_target)

        cap_source = np.percentile(source_std, 99)
        cap_target = np.percentile(target_std, 99)
        cap_overlay = max(cap_source, cap_target)

        # 4.5) Sample the standardized data for plotting
        sample_size = st.number_input(
            "**Select sample size for plotting standardized data**",
            min_value=1,
            max_value=min(len(source_std), len(target_std)),
            value=2000
        )
        sampled_source_std = source_std.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sampled_target_std = target_std.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # 5) Plot individual histograms for the sampled standardized data
        plot_interactive_hist_density(
            sampled_source_std,
            title="Sampled Source Method (Standardized)",
            bins=fd_bins(sampled_source_std),
            x_max_limit=np.percentile(sampled_source_std, 99)
        )
        plot_interactive_hist_density(
            sampled_target_std,
            title="Sampled Target Method (Standardized)",
            bins=fd_bins(sampled_target_std),
            x_max_limit=np.percentile(sampled_target_std, 99)
        )

        # 6) Overlaid histogram for the sampled standardized data
        df_sampled_std = pd.concat([
            sampled_source_std.rename("Source (std)"),
            sampled_target_std.rename("Target (std)")
        ], axis=1).melt(var_name="Method", value_name="Value")
        sampled_cap = np.percentile(df_sampled_std["Value"], 99)
        sampled_bins = max(fd_bins(sampled_source_std), fd_bins(sampled_target_std))
        fig = px.histogram(
            df_sampled_std,
            x="Value",
            color="Method",
            nbins=sampled_bins,
            histnorm="density",
            opacity=0.6,
            barmode="overlay",
            title="Overlaid Density of Sampled Standardized Data",
            color_discrete_map={"Source (std)": "#E74C3C", "Target (std)": "#3498DB"}
        )
        fig.update_xaxes(range=[df_sampled_std["Value"].min(), sampled_cap])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig)

        # 7) KS‐test inputs and execution using full standardized data
        min_size = min(len(source_std), len(target_std))
        st.markdown(f"Minimum available sample size: **{min_size}**")
        subset_size = st.number_input(
            "Subset size for KS test",
            min_value=5,
            max_value=int(min_size),
            value=2000
        )
        alpha = st.number_input(
            "Significance level (alpha)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.005,
            format="%.3f"
        )

        if st.button("Run KS Test"):
            # Set a fixed seed for reproducibility
            np.random.seed(123)
            # Use n1 and n2 for clarity (here both equal subset_size)
            n1 = subset_size
            n2 = subset_size
            src_sample = np.random.choice(source_std, size=n1, replace=False)
            tgt_sample = np.random.choice(target_std, size=n2, replace=False)
            ks_stat, p_value = stats.ks_2samp(src_sample, tgt_sample)

            # Calculate critical D value using the original formula
            c_alpha = np.sqrt(-0.5 * np.log(alpha / 2))
            D_crit = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))

            st.write(f"**KS Statistic (Observed D):** {ks_stat:.4f}")
            st.write(f"**Critical D value (D_crit):** {D_crit:.4f}")
            st.write(f"**p-value:** {p_value:.8f}")

            if ks_stat < D_crit:
                st.success(f"Observed D = {ks_stat:.4f} < Critical D = {D_crit:.4f}: shapes are similar.")
            else:
                st.warning(f"Observed D = {ks_stat:.4f} ≥ Critical D = {D_crit:.4f}: shapes differ.")
            # Informational text using st.info and st.latex
            st.info(
                "The KS statistic (D) is the maximum absolute difference between the two empirical CDFs.\n\n"
                "A significance level (α) was used to compute the critical value using the following formulas:"
            )

            st.latex(r"c(\alpha) = \sqrt{-\frac{1}{2}\ln\left(\frac{\alpha}{2}\right)}")
            st.latex(r"D_{\text{crit}} = c(\alpha)\sqrt{\frac{n_1+n_2}{n_1\,n_2}}")

            st.info(
                "If the observed D is smaller than the critical D value, the null hypothesis (that the samples come "
                "from the same distribution) is not rejected.\n\n"
                "Note: Very large sample sizes can yield a small p-value even for minor differences; hence, the focus "
                "here is on the actual D value."
            )
    else:
        st.warning("Upload both source and target data to proceed.")



with tab4:
    st.markdown("#### :blue[Result transformation formula]")
    # Define the formula as a LaTeX string
    #formula_latex = r"Adjusted Result = L_{target} + \frac{U_{target} - L_{target}}{U_{source} - L_{source}} \times (Result_{source} - L_{source})"
    # Define the formula as a LaTeX string
    formula_latex = r"""
    Adjusted Result = L_{target} + \frac{U_{target} - L_{target}}{U_{source} - L_{source}} \times (Result_{source} - L_{source}) \\
    \text{ } \\
    \text{where:} \\
    \text{ } \\
    L_{target}: \text{Lower percentile (2.5\%) of the data from target method} \\
    U_{target}: \text{Upper percentile (97.5\%) of the data from target method} \\
    L_{source}: \text{Lower percentile (2.5\%) of the data from source method} \\
    U_{source}: \text{Upper percentile (97.5\%) of the data from source method} \\
    Result_{source}: \text{Source method result} 
    """
    # Display the formula using st.latex()
    st.latex(formula_latex)

    def calculate_result_source(l_target, u_target, l_source, u_source, result_source):
        if u_source - l_source == 0:
            st.error("Error: Division by zero. Ensure that U_source and L_source are different values.")
            return None
        
        result_source = l_target + ((result_source - l_source) * (u_target - l_target)/(u_source - l_source) )
        return result_source
    
    st.markdown("#### :blue[Adjusted Result Calculator]")

    # Input fields
    if uploaded_file_target is not None:
        l_target = st.number_input("**:green[Lower Percentile of the Data from Target Method (L target)]**", min_value=0.00000 ,format="%.f", value=percentile_2_5_target)
        u_target = st.number_input("**:green[Upper Percentile of the Data from Target Method (U target)]**", min_value=0.00000 ,format="%.f", value=percentile_97_5_target)
    else:
        l_target = st.number_input("**:green[Lower Percentile of the Data from Target Method (L target)]**", min_value=0.00000 ,format="%.f", value=0.00000)
        u_target = st.number_input("**:green[Upper Percentile of the Data from Target Method (U target)]**", min_value=0.00000 ,format="%.f", value=0.00000)

    if uploaded_file_source is not None:
        l_source = st.number_input("**:green[Lower Percentile of the Data from Source Method (L source)]**", min_value=0.00000 ,format="%.f", value=percentile_2_5_source)
        u_source = st.number_input("**:green[Upper Percentile of the Data from Source Method (U source)]**", min_value=0.00000 ,format="%.f", value=percentile_97_5_source)
    else:
        l_source = st.number_input("**:green[Lower Percentile of the Data from Source Method (L source)]**", min_value=0.00000 ,format="%.f", value=0.00000)
        u_source = st.number_input("**:green[Upper Percentile of the Data from Source Method (U source)]**", min_value=0.00000 ,format="%.f", value=0.00000)
    

    result_source = st.number_input("**:green[Source Method Result (Result target)]**", min_value=0.00000 ,format="%.f", value=0.00000)

    if st.button("Calculate Adjusted Source Method Result"):
        result_source = calculate_result_source(l_target, u_target, l_source, u_source, result_source)
        if result_source is not None:
            st.success(f"Calculated Result Source: {result_source:.4f}")

    st.write("---")
