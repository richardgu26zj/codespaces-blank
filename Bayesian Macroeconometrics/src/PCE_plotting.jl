using CSV, DataFrames, Plots, Plots.Measures, Dates

# Load the dataset
rawdata = CSV.read("data/PCE.csv", DataFrame)
data = Vector(rawdata[1:end-1,2])
data = 400 .*log.(data[2:end] ./data[1:end-1])
#date = range(1959.25,stop = 2025.25,length = length(data))
plt_date = range(1959.25,step = 0.25, length = length(data))
start_date = Date(1959, 4, 1)
date_axis = range(start_date, step = Month(3), length = length(data))
date = [Dates.format(d, "yyyy-")*"Q$(quarterofyear(d))" for d in date_axis]

# Plot the data
gr()
plot(plt_date, data,
     xlabel = "Quarters", 
     ylabel = "Inflation Rate (%)",
     guidefont = font(10, "Palatino"),
     grid =:both,
     legend = false,
     title = "U.S. Inflation Rate (1959-2024)",
     titlefont = font(16, "Palatino"),
     tickfont = font(8, "Palatino"),   
     color = :blue,
     lw = 2.5,
     size = (800, 400),
     xlims = (1959.00, 2025.50),
     #xlimits = (Date(1959,4,1), Date(2025,4,1)),
     left_margin = 10mm,
     bottom_margin = 10mm)

CSV.write("data/PCE_inflation.csv", DataFrame(Date = plt_date, InflationRate = data))
savefig("figures/PCE_inflation.pdf")