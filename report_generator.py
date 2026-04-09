import os
import io
import contextlib
from fpdf import FPDF
from config import PLOTS_DIR

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'AQI Factor Analysis and Forecasting - Unified Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(text_content, output_path="AQI_Project_Report.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    
    # Process text output line by line so it wraps correctly
    for line in text_content.split('\n'):
        # Deal with unicode issues optionally by replacing non-latin chars if using standard fonts
        line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 5, txt=line, ln=True)
    
    # Add plots
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Visualizations", 0, 1, 'C')
    pdf.ln(5)
    
    # Load plots in order
    plots = [
        ("AQI Status Distribution", "aqi_distribution.png"),
        ("Predictor Correlation Heatmap", "correlation_heatmap.png"),
        ("OLS Coefficients", "ols_coefficients.png"),
        ("Residual Diagnostics", "residual_diagnostics.png"),
        ("Ridge Path", "ridge_path.png"),
        ("Lasso Path", "lasso_path.png"),
        ("ACF & PACF", "acf_pacf.png"),
        ("GARCH Forecast", "forecast_garch.png")
    ]
    
    for title, filename in plots:
        filepath = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(filepath):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, title, 0, 1, 'L')
            # Fit image to width
            pdf.image(filepath, w=170)
            pdf.ln(10)
            
            # Put only 1-2 images per page
            # To be safe, add a new page after each image except the last
            if filename != plots[-1][1]:
                pdf.add_page()

    pdf.output(output_path)
    print(f"PDF report generated successfully at {output_path}")

if __name__ == "__main__":
    import main as main_module
    
    print("Running full analysis to generate PDF. This may take a few minutes...")
    
    # Capture stdout
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            main_module.main()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc(file=f)
    
    output_text = f.getvalue()
    generate_pdf(output_text)
