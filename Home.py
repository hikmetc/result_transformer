# Deveoper: Hikmet Can Çubukçu
# Date: 17.02.2025

import streamlit as st
st.set_page_config(page_title="Result Transformer", page_icon="🛠️")
import numpy as np
import pandas as pd
import plotly.express as px


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

tab1, tab2, tab3 = st.tabs(["📖 **Instructions**", "📊 **:green[Distribution and percentiles of uploaded data]**", 
                                "🛠️ **:blue[Result Transformer]**"],)
with tab1:
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
    with st.container():
        st.markdown("#### :blue[The distribution of source method results]")
        if uploaded_file_source is not None:
            # Calculate percentiles
            percentile_2_5_source = np.percentile(analyte_data_source, 2.5)
            percentile_97_5_source = np.percentile(analyte_data_source, 97.5)
            percentile_99_source = np.percentile(analyte_data_source, 99)
            plot_interactive_hist_density(analyte_data_source, x_max_limit = percentile_99_source)
            # Calculate percentiles
            percentile_2_5_source = np.percentile(analyte_data_source, 2.5)
            percentile_97_5_source = np.percentile(analyte_data_source, 97.5)

            st.markdown(f"**:blue[2.5th Percentile:] {percentile_2_5_source}**")
            st.markdown(f"**:blue[97.5th Percentile:] {percentile_97_5_source}**")
        else:
            st.info("Please upload a file to view the distribution and percentiles.")
    st.write("---")

    with st.container():
        st.markdown("#### :blue[The distribution of target method results]")
        if uploaded_file_target is not None:  
            # Calculate percentiles
            percentile_2_5_target = np.percentile(analyte_data_target, 2.5)
            percentile_97_5_target = np.percentile(analyte_data_target, 97.5)
            percentile_99_target = np.percentile(analyte_data_target, 99)        
            
            plot_interactive_hist_density(analyte_data_target, x_max_limit = percentile_99_target)

            st.markdown(f"**:blue[2.5th Percentile:] {percentile_2_5_target}**")
            st.markdown(f"**:blue[97.5th Percentile:] {percentile_97_5_target}**")
        else:
            st.info("Please upload a file to view the distribution and percentiles.")

with tab3:
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
    

    result_source = st.number_input("**:green[Current Source Method Result]**", min_value=0.00000 ,format="%.f", value=0.00000)

    if st.button("Calculate Adjusted Source Method Result"):
        result_source = calculate_result_source(l_target, u_target, l_source, u_source, result_source)
        if result_source is not None:
            st.success(f"**Adjusted source method result: {result_source:.3f}**")

    st.write("---")
