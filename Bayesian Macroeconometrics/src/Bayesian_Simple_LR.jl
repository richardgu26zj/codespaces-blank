using CSV, DataFrames, Plots, Plots.Measures

# Load Data
rawdata = CSV.read("data/PCE.csv",DataFrame);
data = Vector(rawdata[1:end-1,2]);
data = 400 .*log.(data[2:end]./data[1:end-1]);
date = range(1959.25, stop = 2025.25, length = length(data));

# Initialize plotting of data
gr()
plot(date, data, 
     color =:blue,
     lw = 2.5; 
     grid =:both,
     gridalpha = 0.15,
     legend = false,
     xlabel = "Quarters (Q)",
     ylabel = "Inflation Rate (%)",
     guidefont = font(10, "Computer Modern"),
     title = "US Inflation Rate (PCE)",
     titlefont = font(16,"Computer Modern",:bold),
     xlims = (1959.00, 2025.50),
     tickfont = font(8, "Computer Modern")
     )

savefig("figures/inflation.pdf")