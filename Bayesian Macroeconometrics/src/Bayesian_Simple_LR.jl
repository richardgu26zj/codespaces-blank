using CSV, DataFrames, Plots, Plots.Measures

# Load Data
rawdata = CSV.read("data/PCE.csv",DataFrame);
data = Vector(rawdata[1:end-1,2]);
data = 400 .*log.(data[2:end]./data[1:end-1]);
date = range(1959.25, stop = 2025.25, length = length(data));

# Initialize plotting of data
plot(date, data, 
     color =:blue,
     lw = 3; 
     grid =:both,
     legend = false,
     xlabel = "Quarters",
     ylabel = "Inflation Rate (%)",
     title = "US Inflation Rate (PCE)",
     titlefont = font(16, "Times New Roman", :bold),
     )
savefig("figures/inflation.pdf")